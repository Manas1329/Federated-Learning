import os
import time
import pandas as pd
import concurrent.futures
from typing import List, Tuple, Dict, Optional, Union
from flwr.server import Server
from flwr.common import FitRes, EvaluateRes, DisconnectRes, FitIns, EvaluateIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

class AdaptiveServer(Server):
    """
    Custom Flower Server that supports Target vs Minimum clients and
    an adaptive bounded grace period.
    """
    def __init__(
        self,
        client_manager,
        strategy: Strategy,
        target_clients: int,
        min_clients: int,
        initial_grace_period: float = 30.0,
        max_grace_period: float = 45.0,
        round_timeout: float = 300.0,
        suffix: str = "a_pure",
        models_dir: str = "../models"
    ):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.target_clients = target_clients
        self.min_clients = min_clients
        self.initial_grace_period = initial_grace_period
        self.max_grace_period = max_grace_period
        self.round_timeout = round_timeout
        self.suffix = suffix
        self.models_dir = models_dir
        
        self.csv_path = os.path.join(self.models_dir, f"global_model_records_{self.suffix}.csv")
        
        # Historical timing per client: client_name -> EMA
        self.client_timing_ema: Dict[str, float] = {}
        
        # Participation Tracking
        self.participation_stats: Dict[str, Dict[str, int]] = {}

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float]
    ) -> Optional[Tuple[Optional[List[Tuple[ClientProxy, FitRes]]], Dict, Tuple[List, List]]]:
        """
        Overrides the default fit_round to implement the TARGET / GRACE PERIOD logic.
        """
        # 1. Get client instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            print("No clients selected, canceling round.")
            return None

        # Number of clients we actually sent instructions to
        total_selected = len(client_instructions)
        print(f"\n[AdaptiveServer] Round {server_round}: Selected {total_selected} clients.")
        print(f"[AdaptiveServer] Target: {self.target_clients}, Minimum: {self.min_clients}")

        results: List[Tuple[ClientProxy, FitRes]] = []
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
        
        round_start_time = time.perf_counter()
        
        # We will dispatch all clients concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_client = {
                executor.submit(
                    client_proxy.fit, ins, timeout=self.round_timeout, group_id=server_round
                ): (client_proxy, ins)
                for client_proxy, ins in client_instructions
            }

            grace_period_timer = None
            target_reached = False
            
            # Wait for them to complete
            while future_to_client:
                # If we have reached the target and haven't started grace period timer yet
                if len(results) >= self.target_clients and not target_reached:
                    target_reached = True
                    # Calculate adaptive grace period
                    grace_period = self._calculate_adaptive_grace_period(client_instructions, results)
                    grace_period_timer = time.perf_counter() + grace_period
                    print(f"\n[AdaptiveServer] TARGET REACHED ({len(results)}/{self.target_clients}).")
                    print(f"[AdaptiveServer] Starting grace period of {grace_period:.1f} seconds for remaining {total_selected - len(results)} clients...")

                # Determine how long to wait for the next future
                if grace_period_timer is not None:
                    wait_time = grace_period_timer - time.perf_counter()
                    if wait_time <= 0:
                        print(f"\n[AdaptiveServer] Grace period EXPIRED! Canceling remaining {len(future_to_client)} clients.")
                        break # exit while loop, cancel remaining futures
                else:
                    overall_elapsed = time.perf_counter() - round_start_time
                    wait_time = self.round_timeout - overall_elapsed
                    if wait_time <= 0:
                        print(f"\n[AdaptiveServer] Round timeout EXPIRED! Canceling remaining {len(future_to_client)} clients.")
                        break

                try:
                    # Wait for the next future to complete
                    done, _ = concurrent.futures.wait(
                        future_to_client.keys(),
                        timeout=min(wait_time, 1.0), # check frequently
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                except concurrent.futures.TimeoutError:
                    done = set()
                except Exception as e:
                    print(f"Wait exception: {e}")
                    done = set()

                for future in done:
                    client_proxy, ins = future_to_client.pop(future)
                    try:
                        res = future.result()
                        # Flower FitRes vs DisconnectRes
                        if isinstance(res, FitRes):
                            # It's a successful result
                            results.append((client_proxy, res))
                            
                            # Track timing and stats
                            client_name = res.metrics.get("client_name", f"Unknown_{client_proxy.cid}")
                            duration = res.metrics.get("training_duration", 0) # if provided
                            if duration == 0: # fallback
                                duration = time.perf_counter() - round_start_time
                            self._update_client_ema(client_name, duration)
                            self._record_participation(client_name, success=True)
                            
                        else:
                            # It's a failure or disconnect
                            failures.append(res)
                            client_name = f"Unknown_{client_proxy.cid}"
                            self._record_participation(client_name, success=False)
                            
                    except Exception as ex:
                        failures.append(ex)
                        client_name = f"Unknown_{client_proxy.cid}"
                        self._record_participation(client_name, success=False)

            # Cleanup any remaining futures if we broke out due to timeout or grace period
            for future in future_to_client:
                future.cancel()
                # Record these as dropped
                client_proxy, ins = future_to_client[future]
                failures.append(Exception("Grace period or timeout expired"))
                self._record_participation(f"Unknown_{client_proxy.cid}", success=False) # Without res, we only have cid

        print(f"\n[AdaptiveServer] Round {server_round} collection finished.")
        print(f"[AdaptiveServer] Collected {len(results)} successful clients.")
        print(f"[AdaptiveServer] {len(failures)} clients failed or dropped out.")

        # Minimum Client Check
        if len(results) < self.min_clients:
            print(f"[AdaptiveServer] ERROR: Minimum clients ({self.min_clients}) not reached. Only {len(results)} succeeded.")
            print("[AdaptiveServer] Aborting aggregation for this round.")
            return None # Round failed

        # 3. Aggregate results
        aggregated_result = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        if aggregated_result is not None:
            parameters_aggregated, metrics_aggregated = aggregated_result
            self.parameters = parameters_aggregated
            
            # Save the successful global model to CSV
            participating = []
            for _, r in results:
                participating.append(r.metrics.get("client_name", "Unknown"))
            
            self._log_global_model(
                server_round=server_round,
                participating_clients="|".join(participating),
                num_participating=len(results),
                num_dropped=len(failures),
                round_time=time.perf_counter() - round_start_time
            )
            return parameters_aggregated, metrics_aggregated, (results, failures)

        return None, {}, (results, failures)


    def _calculate_adaptive_grace_period(self, client_instructions, results) -> float:
        """
        Calculate grace period based on EMAs of the clients that haven't finished yet.
        """
        # If we have no history, return initial
        if not self.client_timing_ema:
            return self.initial_grace_period
            
        # Try to find the max EMA among expected clients
        expected_max_time = 0
        
        # Note: We can't easily extract client_name from instructions without modifying the strategy
        # or relying on cid mapping. So we just use the global max EMA of any known client as a safety bound.
        if self.client_timing_ema:
            expected_max_time = max(self.client_timing_ema.values())
            
        # The grace period shouldn't exceed max_grace_period, and shouldn't be less than initial
        # A simple heuristic: Wait enough time to cover the slowest known client's EMA + 10s margin
        # relative to the current time, but bounded.
        
        grace = expected_max_time * 0.3 # e.g. give an extra 30% time
        
        if grace < self.initial_grace_period:
            grace = self.initial_grace_period
        elif grace > self.max_grace_period:
            grace = self.max_grace_period
            
        return grace

    def _update_client_ema(self, client_name: str, duration: float):
        if client_name not in self.client_timing_ema:
            self.client_timing_ema[client_name] = duration
        else:
            old_ema = self.client_timing_ema[client_name]
            self.client_timing_ema[client_name] = (0.7 * old_ema) + (0.3 * duration)

    def _record_participation(self, client_name: str, success: bool):
        if client_name.startswith("Unknown"):
            return # Ignore unknown clients for stats
            
        if client_name not in self.participation_stats:
            self.participation_stats[client_name] = {"success": 0, "drop": 0}
            
        if success:
            self.participation_stats[client_name]["success"] += 1
        else:
            self.participation_stats[client_name]["drop"] += 1

    def _log_global_model(self, server_round: int, participating_clients: str, num_participating: int, num_dropped: int, round_time: float):
        if server_round != 10:  # Only log the final aggregated model
            return
            
        shared_csv_path = os.path.join(self.models_dir, "all_global_models_registry.csv")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate L2 norm of the 4 layers to represent the weights in the CSV
        # without crashing the CSV with 8.4 million floating point numbers
        weight_summaries = []
        import numpy as np
        if self.parameters is not None:
            from flwr.common import parameters_to_ndarrays
            ndarrays = parameters_to_ndarrays(self.parameters)
            # CNN has 4 weight tensors + 4 bias tensors = 8 arrays
            for i, arr in enumerate(ndarrays):
                if i % 2 == 0:  # Just the weights, skip biases for the summary
                    norm = np.linalg.norm(arr)
                    weight_summaries.append(f"{norm:.4f}")
        
        layer_norms_str = "[" + ", ".join(weight_summaries) + "]"
        
        df = pd.DataFrame([[
            timestamp,
            f"global_model_{self.suffix}",
            participating_clients,
            num_participating,
            num_dropped,
            self.target_clients,
            self.min_clients,
            round_time,
            layer_norms_str
        ]], columns=[
            "Timestamp",
            "Model_Name",
            "Participating_Clients",
            "Num_Participating",
            "Num_Dropped",
            "Target_Clients",
            "Min_Clients",
            "Final_Round_Time_Sec",
            "Weight_L2_Norms"
        ])
        
        os.makedirs(self.models_dir, exist_ok=True)
        df.to_csv(
            shared_csv_path,
            mode="a",
            header=not os.path.exists(shared_csv_path),
            index=False
        )
