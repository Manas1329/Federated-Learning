import sys
import os
import time
import pytest
from unittest.mock import patch, MagicMock

# Make sure we can import the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def clean_env():
    # Remove all experiment related env variables
    keys_to_remove = ["RUN_MODE", "NETWORK_DROPOUT_ROUND", "ARTIFICIAL_DELAY_SEC", "NONSTATIONARY_DELAYS"]
    for k in keys_to_remove:
        if k in os.environ:
            del os.environ[k]

@patch('federated_healthcare.src.client.time.sleep')
def test_1_normal_mode_no_delay(mock_sleep):
    """Test 1: Normal mode does not activate artificial delay"""
    with patch.dict(os.environ, {"RUN_MODE": "normal"}):
        import federated_healthcare.src.client as client
        import importlib
        importlib.reload(client)
        
        # Call client logic that checks delay
        assert client.RUN_MODE == "normal"
        assert not client._is_experiment_enabled("ARTIFICIAL_DELAY_SEC")

@patch('sys.exit')
def test_2_normal_mode_no_disconnect(mock_exit):
    """Test 2: Normal mode ignores network dropout"""
    with patch.dict(os.environ, {"RUN_MODE": "normal", "NETWORK_DROPOUT_ROUND": "2"}):
        import federated_healthcare.src.client as client
        import importlib
        importlib.reload(client)
        
        assert client.RUN_MODE == "normal"
        assert not client._is_experiment_enabled("NETWORK_DROPOUT_ROUND")
        
@patch('sys.exit')
def test_3_normal_mode_no_forced_failure(mock_exit):
    """Test 3: Normal mode ignores forced failures (simulated via dropouts)"""
    # Since forced failures map to NETWORK_DROPOUT_ROUND or similar logic
    with patch.dict(os.environ, {"RUN_MODE": "normal", "FORCED_FAILURE": "1"}):
        import federated_healthcare.src.client as client
        import importlib
        importlib.reload(client)
        
        assert client.RUN_MODE == "normal"
        assert not client._is_experiment_enabled("FORCED_FAILURE")

def test_4_experiment_mode():
    """Test 4: Experiment mode allows experimental behavior"""
    with patch.dict(os.environ, {"RUN_MODE": "experiment", "ARTIFICIAL_DELAY_SEC": "10"}):
        import federated_healthcare.src.client as client
        import importlib
        importlib.reload(client)
        
        assert client.RUN_MODE == "experiment"
        assert client._is_experiment_enabled("ARTIFICIAL_DELAY_SEC")

def test_5_adaptive_dropout_available():
    """Test 5: AdaptiveDropout is still present and available for normal mode"""
    import federated_healthcare.src.dropout_engine as de
    import federated_healthcare.src.dropout_handler as dh
    import federated_healthcare.src.server as server
    
    assert hasattr(dh, 'AdaptiveServer')
    # Engine is present.

def test_6_config_isolation():
    """Test 6: Experiment-only variables cannot accidentally trigger in normal mode"""
    with patch.dict(os.environ, {"RUN_MODE": "normal", "ARTIFICIAL_DELAY_SEC": "50", "NETWORK_DROPOUT_ROUND": "3"}):
        import federated_healthcare.src.client as client
        import importlib
        importlib.reload(client)
        
        assert client.RUN_MODE == "normal"
        assert not client._is_experiment_enabled("ARTIFICIAL_DELAY_SEC")
        assert not client._is_experiment_enabled("NETWORK_DROPOUT_ROUND")
