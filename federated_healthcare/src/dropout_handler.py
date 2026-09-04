import os
import time
import pandas as pd
import concurrent.futures
import grpc
import threading
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
        models_dir: str = "../models",
        adaptive_dropout_enabled: bool = True,
        fixed_deadline_control: bool = False
    ):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.target_clients = target_clients
        self.min_clients = min_clients
        self.total_rounds = total_rounds
        self.suffix = suffix
        self.models_dir = models_dir
        self.adaptive_dropout_enabled = adaptive_dropout_enabled
        self.fixed_deadline_control = fixed_deadline_control
        
        self.engine = AdaptiveDropoutDecisionEngine(
            hard_deadline=hard_deadline,
            alpha=alpha,
            beta=beta,
            k=k,
            minimum_quorum=min_clients
        )
        
        self.csv_path = os.path.join(self.models_dir, f"global_model_records_{self.suffix}.csv")
        self.participation_stats: Dict[str, Dict[str, int]] = {}
        
        self.busy_clients = set()
        self._busy_lock = threading.Lock()

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

        # Isolated proxy-release wait mechanism to prevent reuse of busy proxies
        wait_start = time.time()
        max_wait = 30.0
        
        while True:
            available_instructions = []
            with self._busy_lock:
                for proxy, ins in client_instructions:
                    if str(proxy.cid) not in self.busy_clients:
                        available_instructions.append((proxy, ins))
            
            if len(available_instructions) >= self.min_clients:
                break
                
            if time.time() - wait_start > max_wait:
                print(f"[AdaptiveServer] Timeout waiting for busy proxies. Proceeding with {len(available_instructions)} available clients.")
                break
                
            time.sleep(1.0)
            
        client_instructions = available_instructions
        
        if not client_instructions:
            print("[AdaptiveServer] No available non-busy clients to select.")
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
            future_to_client = {}
            for client_proxy, ins in client_instructions:
                cid = str(client_proxy.cid)
                
                with self._busy_lock:
                    self.busy_clients.add(cid)
                    
                future = executor.submit(
                    client_proxy.fit, ins, timeout=timeout, group_id=server_round
                )
                
                def make_done_callback(client_id):
                    def cb(fut):
                        with self._busy_lock:
                            if client_id in self.busy_clients:
                                self.busy_clients.remove(client_id)
                    return cb
                    
                future.add_done_callback(make_done_callback(cid))
                future_to_client[future] = (client_proxy, ins)

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

                # Ask the engine if we should keep waiting (if enabled)
                if future_to_client and self.adaptive_dropout_enabled:
                    decisions = self.engine.evaluate_missing_clients()
                    
                    futures_to_cancel = []
                    for future, (client_proxy, _) in future_to_client.items():
                        cid = str(client_proxy.cid)
                        decision = decisions.get(cid)
                        
                        if decision and not decision.should_wait:
                            futures_to_cancel.append((future, client_proxy, decision.reason))
                    
                    # Quorum protection logic
                    max_drops_allowed = max(0, len(results) + len(future_to_client) - self.min_clients)
                    
                    while len(futures_to_cancel) > max_drops_allowed:
                        retained_future, cp, reason = futures_to_cancel.pop()
                        print(f"[AdaptiveServer] Quorum protection prevented dropping client {cp.cid} despite decision: {reason}")
                    
                    for future, client_proxy, reason in futures_to_cancel:
                        future_to_client.pop(future)
                        future.cancel()
                        print(f"[AdaptiveServer] Engine decision: DROP client {client_proxy.cid} ({reason})")
                        self.engine.record_straggler_drop(str(client_proxy.cid))
                        failures.append(Exception("Dropped by Adaptive Dropout Engine"))
                        self._record_participation(str(client_proxy.cid), success=False)
                        
                elif future_to_client and self.fixed_deadline_control:
                    if self.engine.get_elapsed_time() >= self.engine.hard_deadline:
                        if len(results) >= self.min_clients:
                            futures_to_cancel = list(future_to_client.keys())
                            for future in futures_to_cancel:
                                client_proxy, _ = future_to_client.pop(future)
                                future.cancel()
                                print(f"[AdaptiveServer] Fixed deadline: DROP client {client_proxy.cid}")
                                self.engine.record_straggler_drop(str(client_proxy.cid))
                                failures.append(Exception("Dropped by Fixed Deadline Control"))
                                self._record_participation(str(client_proxy.cid), success=False)
                        
                if future_to_client and timeout and self.engine.get_elapsed_time() >= timeout:
                        print(f"\n[AdaptiveServer] Hard round timeout ({timeout}s) expired! Canceling remaining.")
                        break

            for future in list(future_to_client.keys()):
                client_proxy, ins = future_to_client.pop(future)
                future.cancel()
                self.engine.record_straggler_drop(str(client_proxy.cid))
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
