import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Set, Optional

logger = logging.getLogger("DropoutEngine")
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

class ClientState(Enum):
    SELECTED = "SELECTED"
    ACTIVE = "ACTIVE"
    STRAGGLER = "STRAGGLER"
    COMPLETED = "COMPLETED"
    DISCONNECTED = "DISCONNECTED"
    DROPPED = "DROPPED"
    FAILED = "FAILED"
    NOT_REQUIRED_AFTER_QUORUM = "NOT_REQUIRED_AFTER_QUORUM"

@dataclass
class ClientProfile:
    client_id: str
    mu: float = 0.0
    d: float = 0.0
    successful_participations: int = 0
    network_failures: int = 0
    straggler_drops: int = 0
    other_failures: int = 0
    not_required_after_quorum: int = 0
    total_rounds: int = 0
    has_history: bool = False
    last_completion_time: float = 0.0
    last_state: str = ""

    @property
    def reliability(self) -> float:
        total_attempts = (
            self.successful_participations +
            self.network_failures +
            self.straggler_drops +
            self.other_failures
        )
        if total_attempts == 0:
            return 1.0
        return self.successful_participations / total_attempts

@dataclass
class DropoutDecision:
    client_id: str
    state: str
    should_wait: bool
    elapsed_time: float
    expected_completion: float
    variability: float
    predicted_finish_time: float
    remaining_time: float
    reliability: float
    reason: str

class AdaptiveDropoutDecisionEngine:
    def __init__(
        self,
        hard_deadline: float = 60.0,
        alpha: float = 0.30,
        beta: float = 0.30,
        k: float = 1.0,
        minimum_quorum: int = 2
    ):
        self.hard_deadline = hard_deadline
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.minimum_quorum = minimum_quorum
        
        self.profiles: Dict[str, ClientProfile] = {}
        
        # Round state
        self.current_round: int = 0
        self.round_start_time: float = 0.0
        self.active_clients: Dict[str, ClientState] = {}
        self.finalized_clients: Set[str] = set()
        
    def _get_profile(self, client_id: str) -> ClientProfile:
        if client_id not in self.profiles:
            self.profiles[client_id] = ClientProfile(client_id=client_id)
        return self.profiles[client_id]

    def start_round(self, round_id: int, selected_clients: List[str]):
        """Initializes round state and timers."""
        self.current_round = round_id
        self.round_start_time = time.perf_counter()
        self.active_clients = {cid: ClientState.ACTIVE for cid in selected_clients}
        self.finalized_clients = set()
        
        for cid in selected_clients:
            profile = self._get_profile(cid)
            profile.total_rounds += 1
            profile.last_state = ClientState.SELECTED.value
            
        logger.info(f"=== ROUND {round_id} ENGINE START ===")
        logger.info(f"Selected clients: {selected_clients}")

    def _mark_finalized(self, client_id: str, state: ClientState):
        """Marks a client as finalized for the current round, preventing double-counting."""
        if client_id in self.finalized_clients:
            return False
            
        self.active_clients[client_id] = state
        self.finalized_clients.add(client_id)
        
        profile = self._get_profile(client_id)
        profile.last_state = state.value
        return True

    def record_success(self, client_id: str, completion_time: float):
        """Records a successful update and updates the EMA profile."""
        if not self._mark_finalized(client_id, ClientState.COMPLETED):
            return
            
        profile = self._get_profile(client_id)
        a_it = completion_time
        
        if not profile.has_history:
            profile.mu = a_it
            profile.d = 0.0
            profile.has_history = True
        else:
            abs_dev = abs(a_it - profile.mu)
            profile.d = (self.beta * abs_dev) + ((1 - self.beta) * profile.d)
            profile.mu = (self.alpha * a_it) + ((1 - self.alpha) * profile.mu)
            
        profile.successful_participations += 1
        profile.last_completion_time = a_it
        
        logger.info(f"ROUND {self.current_round} | CLIENT {client_id} | STATE COMPLETED | OBSERVED_COMPLETION {a_it:.1f}s | EMA {profile.mu:.1f}s | DEVIATION {profile.d:.1f}s")

    def record_network_failure(self, client_id: str, reason: str = "network disconnect"):
        """Records a confirmed connection drop."""
        if not self._mark_finalized(client_id, ClientState.DISCONNECTED):
            return
            
        profile = self._get_profile(client_id)
        profile.network_failures += 1
        
        logger.info(f"ROUND {self.current_round} | CLIENT {client_id} | STATE DISCONNECTED | REASON {reason} | DECISION DROP")

    def record_straggler_drop(self, client_id: str, reason: str = "deadline violation"):
        """Records a straggler drop."""
        if not self._mark_finalized(client_id, ClientState.DROPPED):
            return
            
        profile = self._get_profile(client_id)
        profile.straggler_drops += 1
        
        logger.info(f"ROUND {self.current_round} | CLIENT {client_id} | STATE DROPPED | REASON {reason}")

    def record_failure(self, client_id: str, reason: str = "generic failure"):
        """Records a generic failure."""
        if not self._mark_finalized(client_id, ClientState.FAILED):
            return
            
        profile = self._get_profile(client_id)
        profile.other_failures += 1
        
        logger.info(f"ROUND {self.current_round} | CLIENT {client_id} | STATE FAILED | REASON {reason}")
        
    def record_not_required(self, client_id: str):
        """Records a client that was no longer required due to quorum being met."""
        if not self._mark_finalized(client_id, ClientState.NOT_REQUIRED_AFTER_QUORUM):
            return
            
        profile = self._get_profile(client_id)
        profile.not_required_after_quorum += 1
        
        logger.info(f"ROUND {self.current_round} | CLIENT {client_id} | STATE NOT_REQUIRED_AFTER_QUORUM | DECISION STOP_WAITING")

    def get_elapsed_time(self) -> float:
        return time.perf_counter() - self.round_start_time

    def predict_completion(self, client_id: str, current_elapsed_time: float) -> DropoutDecision:
        """Evaluates improved absolute finish time formula."""
        profile = self._get_profile(client_id)
        
        if not profile.has_history:
            # Cold Start Policy: Assume it finishes exactly at deadline to prevent unfair drop
            mu = self.hard_deadline
            d = 0.0
        else:
            mu = profile.mu
            d = profile.d
        
        # Absolute predicted finish time
        # T_pred_i,t = max(e_t, mu_i + k * d_i)
        t_safe = mu + (self.k * d)
        t_pred = max(current_elapsed_time, t_safe)
        
        should_wait = t_pred <= self.hard_deadline
        reason = "predicted completion within acceptable deadline" if should_wait else "predicted deadline violation"
        state = ClientState.ACTIVE if should_wait else ClientState.STRAGGLER
        
        return DropoutDecision(
            client_id=client_id,
            state=state.value,
            should_wait=should_wait,
            elapsed_time=current_elapsed_time,
            expected_completion=mu,
            variability=d,
            predicted_finish_time=t_pred,
            remaining_time=max(0.0, t_pred - current_elapsed_time),
            reliability=profile.reliability,
            reason=reason
        )

    def evaluate_missing_clients(self) -> Dict[str, DropoutDecision]:
        """Evaluates all clients that haven't finished or disconnected yet."""
        current_elapsed = self.get_elapsed_time()
        decisions = {}
        
        # Count clients that successfully completed
        definitively_completed = sum(1 for cid, state in self.active_clients.items() if state == ClientState.COMPLETED)
        
        # Pending clients
        pending_clients = [cid for cid, state in self.active_clients.items() if cid not in self.finalized_clients]
        
        for cid in pending_clients:
            decision = self.predict_completion(cid, current_elapsed)
            
            # Quorum protection: If dropping this client means we can't reach quorum, we must wait.
            if not decision.should_wait:
                if definitively_completed + len(pending_clients) <= self.minimum_quorum:
                    decision.should_wait = True
                    decision.reason = "quorum protection (waiting despite missing deadline)"
                    decision.state = ClientState.STRAGGLER.value
            
            # If still not waiting, mark state as DROPPED in decision (but don't finalize here yet, let server do it)
            if not decision.should_wait:
                decision.state = ClientState.DROPPED.value
            
            decisions[cid] = decision
            
            if decision.state == ClientState.STRAGGLER.value or decision.state == ClientState.DROPPED.value:
                log_reason = decision.reason
                logger.info(
                    f"ROUND {self.current_round} | CLIENT {cid} | STATE {decision.state} | "
                    f"ELAPSED {decision.elapsed_time:.1f}s | EXPECTED {decision.expected_completion:.1f}s | "
                    f"DEVIATION {decision.variability:.1f}s | PREDICTED_FINISH {decision.predicted_finish_time:.1f}s | "
                    f"DEADLINE {self.hard_deadline:.1f}s | DECISION {'WAIT' if decision.should_wait else 'DROP'} "
                    f"| REASON {log_reason}"
                )
            
        return decisions

    def finalize_round(self):
        """Cleans up and finalizes the round."""
        logger.info(f"=== ROUND {self.current_round} ENGINE FINALIZE ===")
