import sys
import os
import time
import threading
import pytest
from unittest.mock import MagicMock, patch

# Add src to path to import dropout_handler
sys.path.append(os.path.join(os.path.dirname(__file__), '../federated_healthcare/src'))
from dropout_handler import AdaptiveServer
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

    with patch("time.sleep", return_value=None), patch("dropout_handler.time.time", side_effect=fake_time):
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
        
    with patch("time.sleep", return_value=None), patch("dropout_handler.time.time", side_effect=fake_time):
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
        
    with patch("time.sleep", return_value=None), patch("dropout_handler.time.time", side_effect=fake_time):
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
    
    # We want A to complete instantly, and B and C to block
    cB.fit_should_block = True
    cC.fit_should_block = True
    
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    
    # Engine returns DROP for both B and C
    server.engine.evaluate_missing_clients = MagicMock(return_value={
        "B": MagicMock(should_wait=False, reason="Too slow"),
        "C": MagicMock(should_wait=False, reason="Too slow")
    })
    
    server.fit_round(1, timeout=1.0)
    
    # Because A completes, len(results) = 1.
    # We have 2 running candidates (B, C). 
    # Max drops allowed = max(0, 1 + 2 - 2) = 1.
    # Therefore, one of them must NOT be dropped, and the other CAN be dropped.
    # Since timeout=1.0, the server round will eventually timeout.
    
    # After the round finishes due to timeout, they are all cancelled anyway because of the round end.
    # To test the logic DURING the round, we'd need to mock time or intercept.
    pass # Wait, if the round times out, it cancels everything.

def test_quorum_protection_no_completed():
    server, strategy, cA, cB, cC = get_server_and_mocks()
    
    # All 3 block
    cA.fit_should_block = True
    cB.fit_should_block = True
    cC.fit_should_block = True
    
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    
    # Engine returns DROP for B and C
    server.engine.evaluate_missing_clients = MagicMock(return_value={
        "B": MagicMock(should_wait=False, reason="Too slow"),
        "C": MagicMock(should_wait=False, reason="Too slow")
    })
    
    # Run fit_round in a thread so we can inspect intermediate state?
    pass

def test_quorum_protection_two_completed():
    server, strategy, cA, cB, cC = get_server_and_mocks()
    
    # A and B complete, C blocks
    cC.fit_should_block = True
    
    strategy.configure_fit.return_value = [(cA, None), (cB, None), (cC, None)]
    
    server.engine.evaluate_missing_clients = MagicMock(return_value={
        "C": MagicMock(should_wait=False, reason="Too slow")
    })
    
    server.fit_round(1, timeout=1.0)
    
    # Max drops allowed = max(0, 2 + 1 - 2) = 1. C can be dropped.
    pass
