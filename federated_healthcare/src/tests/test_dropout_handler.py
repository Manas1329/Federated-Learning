import sys
import time
import threading
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_healthcare.src.dropout_handler import AdaptiveServer
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Status, Code, Parameters

class MockClientProxy(ClientProxy):
    def __init__(self, cid):
        super().__init__(cid)
        self.fit_should_block = False
        self.fit_event = threading.Event()
        self.fit_exception = None

    def get_properties(self, ins, timeout, group_id):
        pass
        
    def get_parameters(self, ins, timeout, group_id):
        pass

    def fit(self, ins, timeout, group_id):
        if self.fit_should_block:
            self.fit_event.wait()
        if self.fit_exception:
            raise self.fit_exception
        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=10,
            metrics={}
        )

    def evaluate(self, ins, timeout, group_id):
        pass

    def reconnect(self, ins, timeout, group_id):
        pass

def get_server_and_mocks():
    strategy_mock = MagicMock()
    # Strategy aggregate_fit returns (parameters, metrics)
    strategy_mock.aggregate_fit.return_value = (Parameters(tensors=[], tensor_type=""), {})
    
    server = AdaptiveServer(
        client_manager=MagicMock(),
        strategy=strategy_mock,
        target_clients=3,
        min_clients=2,
        hard_deadline=2.0
    )
    
    client_a = MockClientProxy("A")
    client_b = MockClientProxy("B")
    client_c = MockClientProxy("C")
    
    return server, strategy_mock, client_a, client_b, client_c

def test_one_cancelled_client_remains_busy():
    server, strategy, cA, cB, cC = get_server_and_mocks()
    cC.fit_should_block = True
    
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    server.engine.evaluate_missing_clients = MagicMock(return_value={"C": MagicMock(should_wait=False)})
    
    server.fit_round(1, timeout=1.0)
    
    assert "C" in server.busy_clients
    assert "A" not in server.busy_clients
    assert "B" not in server.busy_clients
    
    # Round 2: Attempt to select all 3 again
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    
    t_val = [0]
    def fake_time():
        res = t_val[0]
        t_val[0] += 31
        return res

    with patch("time.sleep", return_value=None), patch("federated_healthcare.src.dropout_handler.time.time", side_effect=fake_time):
        server.fit_round(2, timeout=1.0)
    
    assert "C" in server.busy_clients
    
    # Allow C to complete
    cC.fit_event.set()
    time.sleep(0.1)
    
    assert "C" not in server.busy_clients

def test_two_cancelled_clients_concurrently():
    server, strategy, cA, cB, cC = get_server_and_mocks()
    cB.fit_should_block = True
    cC.fit_should_block = True
    
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    server.engine.evaluate_missing_clients = MagicMock(return_value={
        "B": MagicMock(should_wait=False),
        "C": MagicMock(should_wait=False)
    })
    
    server.fit_round(1, timeout=1.0)
    
    assert "B" in server.busy_clients
    assert "C" in server.busy_clients
    assert "A" not in server.busy_clients
    
    # A's proxy is healthy
    strategy.configure_fit.return_value = [(cA, None)]
    
    t_val = [0]
    def fake_time():
        res = t_val[0]
        t_val[0] += 31
        return res
        
    with patch("time.sleep", return_value=None), patch("federated_healthcare.src.dropout_handler.time.time", side_effect=fake_time):
        server.fit_round(2, timeout=1.0)
        
    assert "A" not in server.busy_clients
    
    cB.fit_event.set()
    cC.fit_event.set()
    time.sleep(0.1)
    
    assert "B" not in server.busy_clients
    assert "C" not in server.busy_clients

def test_stale_completion_does_not_corrupt():
    server, strategy, cA, cB, cC = get_server_and_mocks()
    cC.fit_should_block = True
    
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    server.engine.evaluate_missing_clients = MagicMock(return_value={"C": MagicMock(should_wait=False)})
    
    server.fit_round(1, timeout=1.0)
    assert "C" in server.busy_clients
    
    # Round 2 with A and B
    strategy.configure_fit.return_value = [(cA, None), (cB, None)]
    
    def delayed_completion():
        time.sleep(0.2)
        cC.fit_event.set()
        
    threading.Thread(target=delayed_completion).start()
    
    t_val = [0]
    def fake_time():
        res = t_val[0]
        t_val[0] += 31
        return res
        
    with patch("time.sleep", return_value=None), patch("federated_healthcare.src.dropout_handler.time.time", side_effect=fake_time):
        server.fit_round(2, timeout=1.0)
    
    time.sleep(0.3)
    assert "C" not in server.busy_clients

def test_abruptly_disconnected_client():
    import grpc
    server, strategy, cA, cB, cC = get_server_and_mocks()
    
    class FakeRpcError(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE
            
    cC.fit_should_block = True
    cC.fit_exception = FakeRpcError()
    
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    server.engine.evaluate_missing_clients = MagicMock(return_value={"C": MagicMock(should_wait=False)})
    
    server.fit_round(1, timeout=1.0)
    assert "C" in server.busy_clients
    
    cC.fit_event.set()
    time.sleep(0.1)
    assert "C" not in server.busy_clients

def test_normal_successful_completion():
    server, strategy, cA, cB, cC = get_server_and_mocks()
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    server.engine.evaluate_missing_clients = MagicMock(return_value={})
    
    server.fit_round(1, timeout=1.0)
    
    assert "A" not in server.busy_clients
    assert "B" not in server.busy_clients
    assert "C" not in server.busy_clients

def test_exception_in_worker_clears_busy_state():
    server, strategy, cA, cB, cC = get_server_and_mocks()
    cC.fit_exception = Exception("Arbitrary worker error")
    
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    server.engine.evaluate_missing_clients = MagicMock(return_value={})
    
    server.fit_round(1, timeout=1.0)
    
    assert "C" not in server.busy_clients

def test_quorum_protection_two_drops_one_completed():
    server, strategy, cA, cB, cC = get_server_and_mocks()
    
    # 1 completed, 2 outstanding, quorum=2
    num_completed = 1
    num_outstanding = 2
    
    # Engine returned DROP for both
    drop_candidates = [
        (MagicMock(), cB, "Too slow"),
        (MagicMock(), cC, "Too slow")
    ]
    
    retained_drops = server._filter_drop_candidates_for_quorum(drop_candidates, num_completed, num_outstanding)
    
    # max_drops = max(0, 1 + 2 - 2) = 1
    # One should be dropped, one should be retained (prevented from dropping)
    assert len(retained_drops) == 1

def test_quorum_protection_no_completed():
    server, strategy, cA, cB, cC = get_server_and_mocks()
    
    # 0 completed, 2 outstanding, quorum=2
    num_completed = 0
    num_outstanding = 2
    
    drop_candidates = [
        (MagicMock(), cB, "Too slow"),
        (MagicMock(), cC, "Too slow")
    ]
    
    retained_drops = server._filter_drop_candidates_for_quorum(drop_candidates, num_completed, num_outstanding)
    
    # max_drops = max(0, 0 + 2 - 2) = 0
    # None should be dropped
    assert len(retained_drops) == 0

def test_quorum_protection_two_completed():
    server, strategy, cA, cB, cC = get_server_and_mocks()
    
    # 2 completed, 1 outstanding, quorum=2
    num_completed = 2
    num_outstanding = 1
    
    drop_candidates = [
        (MagicMock(), cC, "Too slow")
    ]
    
    retained_drops = server._filter_drop_candidates_for_quorum(drop_candidates, num_completed, num_outstanding)
    
    # max_drops = max(0, 2 + 1 - 2) = 1
    # It can be safely dropped
    assert len(retained_drops) == 1

import grpc

class FakeRpcError(grpc.RpcError):
    def __init__(self, code):
        self._code = code
    def code(self):
        return self._code
    def __str__(self):
        return f"FakeRpcError: {self._code}"

def test_timeout_with_quorum_handled_gracefully():
    """Test 1: timeout with quorum"""
    server, strategy, cA, cB, cC = get_server_and_mocks()
    
    # 2 completed, 1 timeout
    cC.fit_should_block = False
    cC.fit_exception = FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
    
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    
    res = server.fit_round(1, timeout=1.0)
    
    # Aggregation should succeed since 2 >= quorum(2)
    assert res is not None
    parameters_aggregated, metrics_aggregated, (results, failures) = res
    assert len(results) == 2
    assert len(failures) == 1
    assert str(failures[0]) == "FLOWER_ROUND_TIMEOUT"

def test_timeout_without_quorum_aborts_cleanly():
    """Test 2: timeout without quorum"""
    server, strategy, cA, cB, cC = get_server_and_mocks()
    
    # 1 completed, 2 timeout
    cB.fit_should_block = False
    cB.fit_exception = FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
    cC.fit_should_block = False
    cC.fit_exception = FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
    
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    
    res = server.fit_round(1, timeout=1.0)
    
    # Aggregation should skip since 1 < quorum(2)
    assert res is None

def test_timeout_classification():
    """Test 3: timeout classification"""
    server, strategy, cA, cB, cC = get_server_and_mocks()
    
    cA.fit_exception = FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
    
    strategy.configure_fit.return_value = [(cA, None)]
    
    server.fit_round(1, timeout=1.0)
    profile = server.engine._get_profile("A")
    assert profile.straggler_drops == 1 # recorded as straggler drop due to timeout
    assert profile.network_failures == 0

def test_unexpected_grpc_error_not_treated_as_timeout():
    """Test 4: unexpected grpc error"""
    server, strategy, cA, cB, cC = get_server_and_mocks()
    
    cA.fit_exception = FakeRpcError(grpc.StatusCode.UNAVAILABLE)
    cB.fit_exception = FakeRpcError(grpc.StatusCode.INTERNAL)
    
    strategy.configure_fit.return_value = [(cA, None), (cB, None)]
    
    server.fit_round(1, timeout=1.0)
    profile_a = server.engine._get_profile("A")
    profile_b = server.engine._get_profile("B")
    
    assert profile_a.straggler_drops == 0
    assert profile_a.network_failures == 1 # UNAVAILABLE maps to network failure
    
    assert profile_b.straggler_drops == 0
    assert profile_b.other_failures == 1 # INTERNAL maps to other failure

def test_multiple_rounds():
    """Test 5: multiple rounds with mixed results"""
    server, strategy, cA, cB, cC = get_server_and_mocks()
    
    # R1: timeout but quorum succeeds (A,B OK; C timeout)
    cC.fit_exception = FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    res1 = server.fit_round(1, timeout=1.0)
    assert res1 is not None
    
    # R2: normal (A,B,C OK)
    cC.fit_exception = None
    res2 = server.fit_round(2, timeout=1.0)
    assert res2 is not None
    
    # R3: timeout without quorum (A OK; B,C timeout)
    cB.fit_exception = FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
    cC.fit_exception = FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
    res3 = server.fit_round(3, timeout=1.0)
    assert res3 is None
    
    # R4: normal (A,B,C OK)
    cB.fit_exception = None
    cC.fit_exception = None
    res4 = server.fit_round(4, timeout=1.0)
    assert res4 is not None
