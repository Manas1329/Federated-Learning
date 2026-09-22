import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ChestCNN

def execute_baseline_unlearning(
    global_model_path,
    target_loader: DataLoader,
    target_client: str = "Hospital_B",
    surviving_loaders: list = None,
    lr=1e-4,
    unlearn_epochs=2,
    finetune_epochs=1
):
    """
    Perform gradient-ascent unlearning on `target_client`'s data.

    Args:
        global_model_path : path to the converged global model .pth file
        target_loader     : DataLoader for the client being unlearned
        target_client     : name string used only for logging (default: 'Hospital_B')
        surviving_loaders : list of DataLoaders for the surviving clients
                            (used for optional fine-tuning, pass [] to skip)
        lr                : learning rate for gradient ascent
        unlearn_epochs    : number of gradient-ascent epochs
        finetune_epochs   : number of fine-tuning epochs on surviving clients
    """
    if surviving_loaders is None:
        surviving_loaders = []

    print("=" * 60)
    print("Initiating Standard Federated Unlearning (Baseline)")
    print(f"Target: Erasing {target_client}'s Contribution")
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
    # Phase 1: Gradient Ascent on Target Client (The Erasure)
    # ==================================================
    print(f"\n[Phase 1] Executing Gradient Ascent on {target_client}'s data...")
    for epoch in range(unlearn_epochs):
        for images, labels in target_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # NEGATIVE loss pushes the model weights away from the target client's features
            (-loss).backward()
            optimizer.step()

    print(f"{target_client} data mathematically unlearned.")

    # ==================================================
    # Phase 2: Fine-Tuning on Surviving Clients (The Recovery)
    # ==================================================
    # Standard FU assumes surviving clients have enough data to recover naturally.
    if finetune_epochs > 0 and surviving_loaders:
        print(f"\n[Phase 2] Executing Standard Fine-Tuning on {len(surviving_loaders)} surviving client(s)...")
        for loader in surviving_loaders:
            for epoch in range(finetune_epochs):
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
    else:
        print("\n[Phase 2] Skipped fine-tuning (finetune_epochs=0 or no surviving loaders provided).")

    print("\nBaseline Unlearning Complete.")
    return model


if __name__ == "__main__":
    # ============================================================
    # BACKWARD-COMPATIBLE ORIGINAL B EXPERIMENT
    # Run this script directly to reproduce the original hardcoded
    # Hospital-B unlearning experiment exactly as before.
    # ============================================================
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

    print("\nExecuting baseline unlearning (original Hospital-B experiment)...")
    # 3. Execute the Baseline Unlearning — same parameters as before
    unlearned_model = execute_baseline_unlearning(
        global_model_path=GLOBAL_MODEL_PATH,
        target_loader=trainloader_b,
        target_client="Hospital_B",
        surviving_loaders=[trainloader_a, trainloader_c],
        lr=5e-3,          # INCREASED from 1e-4. We need to push the weights harder to erase B.
        unlearn_epochs=5, # INCREASED from 2. Force it to forget more thoroughly.
        finetune_epochs=0 # SET TO 0. Pure erasure.
    )

    # 4. Save the resulting model
    save_path = "../models/unlearned_baseline_model.pth"
    import torch
    torch.save(unlearned_model.state_dict(), save_path)
    print(f"\n[SUCCESS] Saved Unlearned Baseline Model to: {save_path}")

