import os
import time
import pandas as pd
import concurrent.futures
import grpc
from typing import List, Tuple, Dict, Optional, Union
from flwr.server import Server
from flwr.common import FitRes, EvaluateRes, DisconnectRes, FitIns, EvaluateIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from dropout_engine import AdaptiveDropoutDecisionEngine, ClientState

class AdaptiveServer(Server):
    """
    Custom Flower Server that integrates the Adaptive Dropout Decision Engine.
    """
    def __init__(
        self,
        client_manager,
        strategy: Strategy,
        target_clients: int,
        min_clients: int,
        total_rounds: int = 10,
        hard_deadline: float = 60.0,
        alpha: float = 0.3,
        beta: float = 0.3,
        k: float = 1.0,
        suffix: str = "a_pure",
        models_dir: str = "../models"
    ):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.target_clients = target_clients
        self.min_clients = min_clients
        self.total_rounds = total_rounds
        self.suffix = suffix
        self.models_dir = models_dir
        
        self.engine = AdaptiveDropoutDecisionEngine(
            hard_deadline=hard_deadline,
            alpha=alpha,
            beta=beta,
            k=k,
            minimum_quorum=min_clients
        )
        
        self.csv_path = os.path.join(self.models_dir, f"global_model_records_{self.suffix}.csv")
        self.participation_stats: Dict[str, Dict[str, int]] = {}

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float]
    ) -> Optional[Tuple[Optional[List[Tuple[ClientProxy, FitRes]]], Dict, Tuple[List, List]]]:
        """
        Overrides the default fit_round to implement the Adaptive Dropout logic.
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

        total_selected = len(client_instructions)
        print(f"\n[AdaptiveServer] Round {server_round}: Selected {total_selected} clients.")
        
        # Initialize engine for the round
        selected_cids = [str(proxy.cid) for proxy, _ in client_instructions]
        self.engine.start_round(server_round, selected_cids)

        results: List[Tuple[ClientProxy, FitRes]] = []
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            future_to_client = {
                executor.submit(
                    client_proxy.fit, ins, timeout=timeout, group_id=server_round
                ): (client_proxy, ins)
                for client_proxy, ins in client_instructions
            }

            while future_to_client:
                # Wait for up to 1 second for any future to complete, to allow periodic engine evaluation
                try:
                    done, _ = concurrent.futures.wait(
                        future_to_client.keys(),
                        timeout=1.0,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                except concurrent.futures.TimeoutError:
                    done = set()
                except Exception as e:
                    print(f"Wait exception: {e}")
                    done = set()

                # Process completed futures
                for future in done:
                    client_proxy, ins = future_to_client.pop(future)
                    cid = str(client_proxy.cid)
                    
                    try:
                        res = future.result()
                        if isinstance(res, FitRes):
                            # SUCCESS
                            completion_time = self.engine.get_elapsed_time()
                            self.engine.record_success(cid, completion_time)
                            results.append((client_proxy, res))
                            self._record_participation(cid, success=True)
                        elif isinstance(res, DisconnectRes):
                            # DISCONNECT
                            self.engine.record_network_failure(cid)
                            failures.append(res)
                            self._record_participation(cid, success=False)
                        else:
                            # OTHER FAILURE
                            self.engine.record_failure(cid)
                            failures.append(res)
                            self._record_participation(cid, success=False)
                            
                    except Exception as ex:
                        is_network_failure = False
                        if isinstance(ex, grpc.RpcError):
                            # True gRPC error can be mapped to connection lost
                            if ex.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED):
                                is_network_failure = True
                        
                        if is_network_failure:
                            self.engine.record_network_failure(cid, reason=f"grpc code: {ex.code().name}")
                        else:
                            self.engine.record_failure(cid, reason=str(ex))
                            
                        failures.append(ex)
                        self._record_participation(cid, success=False)

                # If all expected target clients have completed, we can stop waiting for the rest
                if len(results) >= self.target_clients:
                    print(f"\n[AdaptiveServer] TARGET REACHED ({len(results)}/{self.target_clients}).")
                    for cid in [str(cp.cid) for future, (cp, _) in future_to_client.items()]:
                        self.engine.record_not_required(cid)
                    break

                # Ask the engine if we should keep waiting
                if future_to_client:
                    decisions = self.engine.evaluate_missing_clients()
                    
                    futures_to_cancel = []
                    for future, (client_proxy, _) in future_to_client.items():
                        cid = str(client_proxy.cid)
                        decision = decisions.get(cid)
                        
                        if decision and not decision.should_wait:
                            print(f"[AdaptiveServer] Engine decision: DROP client {cid} ({decision.reason})")
                            futures_to_cancel.append(future)
                    
                    for future in futures_to_cancel:
                        client_proxy, _ = future_to_client.pop(future)
                        future.cancel()
                        self.engine.record_straggler_drop(str(client_proxy.cid))
                        failures.append(Exception("Dropped by Adaptive Dropout Engine"))
                        self._record_participation(str(client_proxy.cid), success=False)
                        
                    if timeout and self.engine.get_elapsed_time() >= timeout:
                        print(f"\n[AdaptiveServer] Hard round timeout ({timeout}s) expired! Canceling remaining.")
                        break

            for future in list(future_to_client.keys()):
                client_proxy, ins = future_to_client.pop(future)
                future.cancel()
                self.engine.record_failure(str(client_proxy.cid), reason="Round ended or timed out")
                failures.append(Exception("Round ended or timed out"))
                self._record_participation(str(client_proxy.cid), success=False)
        
        finally:
            # shutdown without waiting so stragglers don't block the server round
            executor.shutdown(wait=False, cancel_futures=True)

        self.engine.finalize_round()

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
            participating = [str(r[0].cid) for r in results]
            
            self._log_global_model(
                server_round=server_round,
                participating_clients="|".join(participating),
                num_participating=len(results),
                num_dropped=len(failures),
                round_time=self.engine.get_elapsed_time()
            )
            return parameters_aggregated, metrics_aggregated, (results, failures)

        return None, {}, (results, failures)

    def _record_participation(self, client_id: str, success: bool):
        if client_id not in self.participation_stats:
            self.participation_stats[client_id] = {"success": 0, "drop": 0}
            
        if success:
            self.participation_stats[client_id]["success"] += 1
        else:
            self.participation_stats[client_id]["drop"] += 1

    def _log_global_model(self, server_round: int, participating_clients: str, num_participating: int, num_dropped: int, round_time: float):
        if server_round != self.total_rounds:  # Only log the final aggregated model
            return
            
        shared_csv_path = os.path.join(self.models_dir, "all_global_models_registry.csv")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        weight_summaries = []
        import numpy as np
        if self.parameters is not None:
            from flwr.common import parameters_to_ndarrays
            ndarrays = parameters_to_ndarrays(self.parameters)
            for i, arr in enumerate(ndarrays):
                if i % 2 == 0:
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
