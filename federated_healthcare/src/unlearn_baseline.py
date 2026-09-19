import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ChestCNN

def execute_baseline_unlearning(
    global_model_path, 
    hospital_b_loader: DataLoader, 
    hospital_a_loader: DataLoader, 
    hospital_c_loader: DataLoader, 
    lr=1e-4, 
    unlearn_epochs=2, 
    finetune_epochs=1
):
    print("=" * 60)
    print("Initiating Standard Federated Unlearning (Baseline)")
    print("Target: Erasing Hospital B's Contribution")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the converged global FL model
    model = ChestCNN().to(device)
    model.load_state_dict(torch.load(global_model_path, map_location=device))
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # ==================================================
    # Phase 1: Gradient Ascent on Hospital B (The Erasure)
    # ==================================================
    print("\n[Phase 1] Executing Gradient Ascent on Hospital B's data...")
    for epoch in range(unlearn_epochs):
        for images, labels in hospital_b_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # NEGATIVE loss pushes the model weights away from Hospital B's features
            (-loss).backward()
            optimizer.step()
            
    print("Hospital B data mathematically unlearned.")

    # ==================================================
    # Phase 2: Fine-Tuning on Hospitals A & C (The Recovery)
    # ==================================================
    # Standard FU assumes A and C have enough data to recover the model naturally.
    print("\n[Phase 2] Executing Standard Fine-Tuning on Surviving Clients (A & C)...")
    
    # Simulate Hospital A local training
    for epoch in range(finetune_epochs):
        for images, labels in hospital_a_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Simulate Hospital C local training
    for epoch in range(finetune_epochs):
        for images, labels in hospital_c_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print("\nBaseline Unlearning Complete.")
    return model


if __name__ == "__main__":
    print("Starting unlearn_baseline.py script...")
    
    # 1. Path to your fully trained global model
    # (Adjust this if your models are saved elsewhere)
    GLOBAL_MODEL_PATH = "../models/global_model_a_pure.pth" 
    
    # 2. Import the data loader function from your client script
    # This matches the function used in your client.py
    from client import load_partitions
    
    print("Loading Hospital datasets...")
    
    # In your client.py, load_partitions returns (trainloader, testloader, num_examples)
    # We only need the trainloader for the unlearning process, so we use _ to ignore the rest
    trainloader_a, _, _ = load_partitions(client_id="Hospital_A")
    trainloader_b, _, _ = load_partitions(client_id="Hospital_B")
    trainloader_c, _, _ = load_partitions(client_id="Hospital_C")
    
    print("\nExecuting baseline unlearning...")
    # 3. Execute the Baseline Unlearning
    # unlearned_model = execute_baseline_unlearning(
    #     global_model_path=GLOBAL_MODEL_PATH,
    #     hospital_b_loader=trainloader_b,
    #     hospital_a_loader=trainloader_a,
    #     hospital_c_loader=trainloader_c,
    #     lr=1e-4,              
    #     unlearn_epochs=2,     
    #     finetune_epochs=1     
    # )
    # 3. Execute the Baseline Unlearning
    unlearned_model = execute_baseline_unlearning(
        global_model_path=GLOBAL_MODEL_PATH,
        hospital_b_loader=trainloader_b,
        hospital_a_loader=trainloader_a,
        hospital_c_loader=trainloader_c,
        lr=5e-3,              # INCREASED from 1e-4. We need to push the weights harder to erase B.
        unlearn_epochs=5,     # INCREASED from 2. Force it to forget more thoroughly.
        finetune_epochs=0     # SET TO 0. Pure erasure. Let's see what the model looks like *before* A and C try to fix it naturally.
    )
    
    # 4. Save the resulting model
    save_path = "../models/unlearned_baseline_model.pth"
    import torch
    torch.save(unlearned_model.state_dict(), save_path)
    print(f"\n[SUCCESS] Saved Unlearned Baseline Model to: {save_path}")

