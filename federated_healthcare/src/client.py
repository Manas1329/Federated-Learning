import sys
import os

# Ensure 'src' package is importable regardless of where the script is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flwr as fl
import torch
from collections import OrderedDict
from model import ChestCNN, train, test
from utils import load_hospital_data

# Load local parameters
DATA_PATH = os.environ.get("DATA_PATH", "./data/hospital_A")
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "localhost:8080")

# Initialize Local Model
net = ChestCNN()

# Load actual data using the utility
trainloader, testloader = load_hospital_data(DATA_PATH)

class HospitalClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Train model locally for 2 epochs
        train(net, trainloader, epochs=2)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        print(f"Local Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=HospitalClient())
