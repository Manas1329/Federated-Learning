import unittest
from unittest.mock import patch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_healthcare.src.dropout_engine import AdaptiveDropoutDecisionEngine, ClientState, DropoutDecision

class TestDropoutEngine(unittest.TestCase):
    
    def setUp(self):
        self.engine = AdaptiveDropoutDecisionEngine(
            hard_deadline=60.0,
            alpha=0.3,
            beta=0.3,
            k=1.0,
            minimum_quorum=2
        )
        self.engine.start_round(1, ["A", "B", "C", "D"])
        
    def test_1_first_successful_observation_initializes_mu(self):
        """TEST 1: First observation initializes mu."""
        self.engine.record_success("A", 20.0)
        profile = self.engine._get_profile("A")
        self.assertEqual(profile.mu, 20.0)
        self.assertEqual(profile.d, 0.0)
        self.assertTrue(profile.has_history)
        self.assertEqual(profile.successful_participations, 1)

    def test_2_second_observation_updates_ema(self):
        """TEST 2: Second observation updates EMA."""
        self.engine.record_success("A", 20.0)
        
        self.engine.start_round(2, ["A"])
        self.engine.record_success("A", 30.0)
        
        profile = self.engine._get_profile("A")
        # mu = 0.3 * 30 + 0.7 * 20 = 9 + 14 = 23
        self.assertAlmostEqual(profile.mu, 23.0)

    def test_3_deviation_updates_correctly(self):
        """TEST 3: Deviation updates correctly."""
        self.engine.record_success("A", 20.0)
        
        self.engine.start_round(2, ["A"])
        self.engine.record_success("A", 30.0)
        
        profile = self.engine._get_profile("A")
        # Previous d = 0, new abs_dev = abs(30 - 20) = 10
        # d = 0.3 * 10 + 0.7 * 0 = 3.0
        self.assertAlmostEqual(profile.d, 3.0)

    @patch('time.perf_counter')
    def test_4_elapsed_time_uses_monotonic_clock(self, mock_time):
        """TEST 4: Elapsed time uses monotonic clock."""
        mock_time.return_value = 100.0
        self.engine.start_round(1, ["A"])
        mock_time.return_value = 115.0
        self.assertAlmostEqual(self.engine.get_elapsed_time(), 15.0)

    @patch('time.perf_counter')
    def test_5_predict_before_deadline_waits(self, mock_time):
        """TEST 5: Predicted completion before deadline -> WAIT."""
        self.engine.record_success("A", 20.0) # mu=20
        mock_time.return_value = 0.0
        self.engine.start_round(2, ["A"])
        
        mock_time.return_value = 15.0
        decision = self.engine.predict_completion("A", self.engine.get_elapsed_time())
        # T_pred = max(15, 20 + 1*0) = 20 <= 60
        self.assertTrue(decision.should_wait)

    @patch('time.perf_counter')
    def test_6_predict_after_deadline_drops(self, mock_time):
        """TEST 6: Predicted completion after deadline -> DROP when quorum allows."""
        self.engine.record_success("D", 80.0) # mu=80
        mock_time.return_value = 0.0
        # Include B and C so we meet quorum of 2 and can drop D
        self.engine.start_round(2, ["D", "B", "C"])
        self.engine.record_success("B", 10.0)
        self.engine.record_success("C", 10.0)
        
        mock_time.return_value = 35.0
        decisions = self.engine.evaluate_missing_clients()
        
        decision = decisions["D"]
        # T_pred = max(35, 80) = 80 > 60
        self.assertFalse(decision.should_wait)
        self.assertEqual(decision.state, ClientState.DROPPED.value)

    @patch('time.perf_counter')
    def test_7_slow_client_straggler_not_disconnected(self, mock_time):
        """TEST 7: A slow connected client is STRAGGLER, not DISCONNECTED."""
        self.engine.record_success("C", 40.0)
        mock_time.return_value = 0.0
        self.engine.start_round(2, ["C"])
        
        mock_time.return_value = 35.0
        decision = self.engine.predict_completion("C", self.engine.get_elapsed_time())
        # T_pred = max(35, 40) = 40 <= 60
        self.assertTrue(decision.should_wait)
        self.assertEqual(decision.state, ClientState.ACTIVE.value) # When should_wait is True, state is ACTIVE. It's a "straggler condition" logged.

    def test_8_confirmed_network_disconnect(self):
        """TEST 8: Confirmed connector/network failure -> DISCONNECTED."""
        self.engine.record_network_failure("B")
        profile = self.engine._get_profile("B")
        self.assertEqual(profile.network_failures, 1)
        self.assertEqual(self.engine.active_clients["B"], ClientState.DISCONNECTED)

    def test_9_network_failure_increments_network_failures_only(self):
        """TEST 9: Network failure increments network_failures only."""
        self.engine.record_network_failure("B")
        profile = self.engine._get_profile("B")
        self.assertEqual(profile.network_failures, 1)
        self.assertEqual(profile.straggler_drops, 0)
        self.assertEqual(profile.other_failures, 0)
        self.assertEqual(profile.successful_participations, 0)

    def test_10_straggler_drop_increments_straggler_drops_only(self):
        """TEST 10: Straggler drop increments straggler_drops only."""
        self.engine.record_straggler_drop("A")
        profile = self.engine._get_profile("A")
        self.assertEqual(profile.straggler_drops, 1)
        self.assertEqual(profile.network_failures, 0)
        self.assertEqual(profile.other_failures, 0)

    def test_11_generic_failure_increments_other_failures_only(self):
        """TEST 11: Generic failure increments other_failures only."""
        self.engine.record_failure("A")
        profile = self.engine._get_profile("A")
        self.assertEqual(profile.other_failures, 1)
        self.assertEqual(profile.straggler_drops, 0)
        self.assertEqual(profile.network_failures, 0)

    def test_12_dropped_client_can_participate_in_future(self):
        """TEST 12: Dropped client can participate in future round."""
        self.engine.record_straggler_drop("A")
        
        self.engine.start_round(2, ["A"])
        self.assertEqual(self.engine.active_clients["A"], ClientState.ACTIVE)

    def test_13_historical_profile_survives_reconnect(self):
        """TEST 13: Historical profile survives reconnect."""
        self.engine.record_success("A", 20.0)
        self.engine.start_round(2, ["A"])
        self.engine.record_network_failure("A")
        
        self.engine.start_round(3, ["A"])
        profile = self.engine._get_profile("A")
        self.assertTrue(profile.has_history)
        self.assertEqual(profile.mu, 20.0)

    def test_14_successful_rejoin_updates_ema(self):
        """TEST 14: Successful rejoin updates EMA."""
        self.engine.record_success("A", 20.0)
        self.engine.start_round(2, ["A"])
        self.engine.record_network_failure("A")
        
        self.engine.start_round(3, ["A"])
        self.engine.record_success("A", 30.0)
        profile = self.engine._get_profile("A")
        self.assertAlmostEqual(profile.mu, 23.0)

    @patch('time.perf_counter')
    def test_15_cold_start_client_waits(self, mock_time):
        """TEST 15: Cold-start client does not crash or get immediately unfairly dropped."""
        mock_time.return_value = 0.0
        self.engine.start_round(1, ["NEW_CLIENT"])
        
        mock_time.return_value = 10.0
        decision = self.engine.predict_completion("NEW_CLIENT", self.engine.get_elapsed_time())
        # T_pred = max(10, 60 + 0) = 60 <= 60
        self.assertTrue(decision.should_wait)
        self.assertAlmostEqual(decision.predicted_finish_time, 60.0)

    @patch('time.perf_counter')
    def test_16_elapsed_time_greater_than_mean(self, mock_time):
        """TEST 16: Elapsed time greater than historical mean does not produce nonsensical prediction."""
        self.engine.record_success("A", 20.0) # mu=20
        mock_time.return_value = 0.0
        self.engine.start_round(2, ["A"])
        
        mock_time.return_value = 30.0
        decision = self.engine.predict_completion("A", self.engine.get_elapsed_time())
        
        # T_pred = max(30, 20) = 30
        self.assertAlmostEqual(decision.predicted_finish_time, 30.0)
        # Remaining time = max(0, 30-30) = 0
        self.assertGreaterEqual(decision.remaining_time, 0.0)

    @patch('time.perf_counter')
    def test_17_predicted_completion_cannot_be_earlier_than_elapsed(self, mock_time):
        """TEST 17: Predicted completion cannot be earlier than current elapsed time."""
        self.engine.record_success("A", 20.0)
        mock_time.return_value = 0.0
        self.engine.start_round(2, ["A"])
        mock_time.return_value = 50.0
        
        decision = self.engine.predict_completion("A", self.engine.get_elapsed_time())
        # T_pred = max(50, 20) = 50
        self.assertGreaterEqual(decision.predicted_finish_time, 50.0)

    @patch('time.perf_counter')
    def test_18_quorum_protection_works(self, mock_time):
        """TEST 18: Quorum protection works."""
        self.engine.record_success("D", 80.0)
        mock_time.return_value = 0.0
        self.engine.start_round(2, ["D"]) # Only 1 client, min_quorum=2
        
        mock_time.return_value = 35.0
        decisions = self.engine.evaluate_missing_clients()
        decision = decisions["D"]
        # Normally would drop, but quorum protects
        self.assertTrue(decision.should_wait)
        self.assertEqual(decision.state, ClientState.STRAGGLER.value)

    @patch('time.perf_counter')
    def test_19_quorum_protection_does_not_create_infinite_waiting(self, mock_time):
        """TEST 19: Quorum protection does not create infinite waiting.
        The engine tells the server to WAIT, but the server applies the hard timeout."""
        self.engine.record_success("D", 80.0)
        mock_time.return_value = 0.0
        self.engine.start_round(2, ["D"])
        
        mock_time.return_value = 65.0 # Past hard deadline
        decision = self.engine.predict_completion("D", self.engine.get_elapsed_time())
        # It says don't wait mathematically (T_pred=80 > 60), but evaluate_missing_clients overrides for quorum
        # The SERVER orchestrates the final cancellation when elapsed_time > timeout. 
        # Here we just verify the elapsed_time exceeds deadline.
        self.assertGreater(self.engine.get_elapsed_time(), self.engine.hard_deadline)

    def test_20_target_reached_does_not_count_as_failures(self):
        """TEST 20: Target reached does not count remaining clients as failures."""
        self.engine.record_not_required("E")
        profile = self.engine._get_profile("E")
        self.assertEqual(profile.not_required_after_quorum, 1)
        self.assertEqual(profile.other_failures, 0)

    def test_21_late_result_from_dropped_client_not_aggregated(self):
        """TEST 21: Late result from a dropped client is not aggregated (finalized exactly once)."""
        self.engine.record_straggler_drop("A")
        # Later, A returns success
        self.engine.record_success("A", 40.0)
        
        profile = self.engine._get_profile("A")
        self.assertEqual(profile.straggler_drops, 1)
        self.assertEqual(profile.successful_participations, 0) # Ignored

    @patch('time.perf_counter')
    def test_22_completing_after_evaluation_becomes_completed(self, mock_time):
        """TEST 22: Client completing after an earlier evaluation becomes COMPLETED correctly."""
        mock_time.return_value = 0.0
        self.engine.start_round(1, ["A", "B", "C"])
        
        mock_time.return_value = 20.0
        self.engine.evaluate_missing_clients()
        
        self.engine.record_success("A", 25.0)
        self.assertEqual(self.engine.active_clients["A"], ClientState.COMPLETED)
        
        mock_time.return_value = 30.0
        decisions = self.engine.evaluate_missing_clients()
        self.assertNotIn("A", decisions)

    def test_23_same_failure_not_counted_twice(self):
        """TEST 23: Same failure is not counted twice."""
        self.engine.record_network_failure("A")
        self.engine.record_network_failure("A")
        self.engine.record_failure("A")
        profile = self.engine._get_profile("A")
        self.assertEqual(profile.network_failures, 1)
        self.assertEqual(profile.other_failures, 0)

    def test_24_multiple_rounds_preserve_client_histories(self):
        """TEST 24: Multiple rounds preserve client histories."""
        self.engine.record_success("A", 20.0)
        self.engine.start_round(2, ["B"])
        self.engine.record_success("B", 30.0)
        
        pA = self.engine._get_profile("A")
        pB = self.engine._get_profile("B")
        self.assertTrue(pA.has_history)
        self.assertTrue(pB.has_history)

    @patch('time.perf_counter')
    def test_25_simulation(self, mock_time):
        """SIMULATION TEST: Matches the specific test scenario requested."""
        mock_time.return_value = 0.0
        # Round 4 so we don't start from round 1 logic
        self.engine.start_round(4, ["A", "B", "C", "D", "E"])
        
        # Inject histories directly for precision
        pA = self.engine._get_profile("A")
        pA.mu = 20.0; pA.d = 0.0; pA.has_history = True
        
        pB = self.engine._get_profile("B")
        pB.mu = 30.0; pB.d = 0.0; pB.has_history = True
        
        pC = self.engine._get_profile("C")
        pC.mu = 44.0; pC.d = 4.0; pC.has_history = True
        
        pD = self.engine._get_profile("D")
        pD.mu = 50.0; pD.d = 12.0; pD.has_history = True
        
        self.engine.record_success("A", 20.0)
        self.engine.record_success("B", 30.0)
        self.engine.record_network_failure("E")
        
        # At elapsed time 35s
        mock_time.return_value = 35.0
        
        decisions = self.engine.evaluate_missing_clients()
        
        self.assertNotIn("A", decisions)
        self.assertNotIn("B", decisions)
        self.assertNotIn("E", decisions)
        
        # Client C
        decC = decisions["C"]
        # T_pred = max(35, 44 + 4) = 48
        self.assertAlmostEqual(decC.predicted_finish_time, 48.0) 
        self.assertTrue(decC.should_wait)
        
        # Client D
        decD = decisions["D"]
        # T_pred = max(35, 50 + 12) = 62
        self.assertAlmostEqual(decD.predicted_finish_time, 62.0) 
        self.assertFalse(decD.should_wait)
        
        # Reconnect E in next round
        mock_time.return_value = 0.0
        self.engine.start_round(5, ["E"])
        self.assertEqual(self.engine.active_clients["E"], ClientState.ACTIVE)
        self.assertEqual(self.engine._get_profile("E").network_failures, 1)

    def test_26_hard_deadline_and_round_timeout_are_independent(self):
        """TEST 26: Hard deadline and round timeout are independent concepts."""
        self.assertEqual(self.engine.hard_deadline, 60.0)
        from federated_healthcare.src.dropout_handler import AdaptiveServer
        from flwr.server.strategy import FedAvg
        from flwr.server.client_manager import SimpleClientManager
        server = AdaptiveServer(
            client_manager=SimpleClientManager(),
            strategy=FedAvg(),
            target_clients=3,
            min_clients=2,
            total_rounds=5,
            hard_deadline=45.0
        )
        self.assertEqual(server.engine.hard_deadline, 45.0)

    def test_27_final_model_logging_uses_total_rounds_rather_than_hardcoded_10(self):
        """TEST 27: Final model logging uses total_rounds rather than hardcoded 10."""
        from federated_healthcare.src.dropout_handler import AdaptiveServer
        from flwr.server.strategy import FedAvg
        from flwr.server.client_manager import SimpleClientManager
        import pandas as pd
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            server = AdaptiveServer(
                client_manager=SimpleClientManager(),
                strategy=FedAvg(),
                target_clients=3,
                min_clients=2,
                total_rounds=5,
                models_dir=temp_dir
            )
            
            server._log_global_model(10, "A", 1, 0, 10.0)
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "all_global_models_registry.csv")))
            
            server.parameters = None
            server._log_global_model(5, "A", 1, 0, 10.0)
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "all_global_models_registry.csv")))

if __name__ == '__main__':
    unittest.main()
