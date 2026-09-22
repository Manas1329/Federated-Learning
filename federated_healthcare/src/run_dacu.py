import torch
import torch.nn as nn
import numpy as np
from model import ChestCNN
from client import load_partitions

def calculate_dacu_weights(surviving_counts, tau_safe=0.50, alpha=3.0):
    """
    Novelty Step 1 & 2: Audit distribution and calculate compensatory weight.
    """
    total_normal = sum(c["NORMAL"] for c in surviving_counts.values())
    total_pneumonia = sum(c["PNEUMONIA"] for c in surviving_counts.values())
    total = total_normal + total_pneumonia
    
    gamma_pneumonia = total_pneumonia / total
    print(f"\n[DACU Audit] Surviving Pneumonia Ratio: {gamma_pneumonia*100:.2f}% (Safety Threshold: {tau_safe*100:.2f}%)")
    
    if gamma_pneumonia < tau_safe:
        # The DACU Logarithmic Compensation Formula
        w = 1.0 + alpha * np.log(tau_safe / gamma_pneumonia)
        print(f"[DACU Trigger] Shortage detected. Transmitting compensation weight: {w:.4f}")
        return torch.tensor([1.0, float(w)], dtype=torch.float32)
        
    return torch.tensor([1.0, 1.0], dtype=torch.float32)

if __name__ == "__main__":
    print("="*60)
    print("Initiating DACU Protocol: Distribution-Aware Recovery")
    print("="*60)
    
    # 1. Server audits metadata (NOT raw images)
    # Hospital A: 1000 Normal, 250 Pneumonia
    # Hospital C: 500 Normal, 500 Pneumonia
    surviving_metadata = {
        "Hospital_A": {"NORMAL": 1000, "PNEUMONIA": 250},
        "Hospital_C": {"NORMAL": 500,  "PNEUMONIA": 500}
    }
    
    dacu_weights = calculate_dacu_weights(surviving_metadata, tau_safe=0.50, alpha=3.0)
    
    # 2. Load the COLLAPSED model (the dangerous one)
    model = ChestCNN()
    model.load_state_dict(torch.load("../models/unlearned_baseline_model.pth"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    # 3. Apply DACU Hardware Reconfiguration (Inject dynamic weights)
    criterion = nn.CrossEntropyLoss(weight=dacu_weights.to(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    print("\nLoading surviving client datasets (A & C)...")
    trainloader_a, _, _ = load_partitions(client_id="Hospital_A")
    trainloader_c, _, _ = load_partitions(client_id="Hospital_C")
    
    print("\nExecuting DACU Compensatory Training on A and C...")
    epochs_per_client = 2
    
    # Client A Recovery execution
    for epoch in range(epochs_per_client):
        for images, labels in trainloader_a:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            
    # Client C Recovery execution
    for epoch in range(epochs_per_client):
        for images, labels in trainloader_c:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            
    # 4. Save the Compensated Model for the Verification Gate
    save_path = "../models/dacu_compensated_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] DACU Model saved to {save_path}")