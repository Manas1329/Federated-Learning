import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model import ChestCNN
from client import load_partitions

def evaluate_model(model_path, test_loaders):
    model = ChestCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for loader in test_loaders:
            for images, labels in loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
    acc = accuracy_score(all_labels, all_preds)
    # Assuming PNEUMONIA is class 1 and NORMAL is class 0
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, prec, rec, f1, cm


def evaluate_model_by_clients(model_path, client_ids, map_location=None):
    """
    Dynamic variant: loads test data for each client_id in `client_ids`
    and evaluates the model at `model_path`.

    Args:
        model_path  : str/Path — path to a .pth checkpoint
        client_ids  : list of str — e.g. ["Hospital_A", "Hospital_C"]
        map_location: torch device string, or None for auto-detect

    Returns:
        (acc, precision, recall, f1, confusion_matrix)
    """
    from utils import load_partitions
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    device = map_location or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChestCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    for cid in client_ids:
        _, testloader, _ = load_partitions(client_id=cid)
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)

    return acc, prec, rec, f1, cm

if __name__ == "__main__":
    print("Loading test datasets for evaluation...")
    
    # We only need the test loaders to evaluate the global performance
    _, testloader_a, _ = load_partitions(client_id="Hospital_A")
    _, testloader_b, _ = load_partitions(client_id="Hospital_B")
    _, testloader_c, _ = load_partitions(client_id="Hospital_C")
    
    global_test_loaders = [testloader_a, testloader_b, testloader_c]
    
    print("\n" + "="*50)
    print(" 1. EVALUATING ORIGINAL FL MODEL (A + B + C)")
    print("="*50)
    acc1, prec1, rec1, f11, cm1 = evaluate_model("../models/global_model_a_pure.pth", global_test_loaders)
    print(f"Accuracy : {acc1:.4f}")
    print(f"Precision: {prec1:.4f}")
    print(f"Recall   : {rec1:.4f}  <-- Baseline target")
    print(f"F1 Score : {f11:.4f}")
    print(f"Confusion Matrix:\n{cm1}")
    
    print("\n" + "="*50)
    print(" 2. EVALUATING STANDARD UNLEARNED MODEL (B Erased)")
    print("="*50)
    acc2, prec2, rec2, f12, cm2 = evaluate_model("../models/unlearned_baseline_model.pth", global_test_loaders)
    print(f"Accuracy : {acc2:.4f}")
    print(f"Precision: {prec2:.4f}")
    print(f"Recall   : {rec2:.4f}  <-- THIS SHOULD COLLAPSE")
    print(f"F1 Score : {f12:.4f}")
    print(f"Confusion Matrix:\n{cm2}")


    print("\n" + "="*50)
    print(" 3. EVALUATING DACM COMPENSATED MODEL")
    print("="*50)
    import os
    _dacm_model_path = "../models/dacm_compensated_model.pth" if os.path.exists("../models/dacm_compensated_model.pth") else "../models/dacu_compensated_model.pth"
    acc3, prec3, rec3, f13, cm3 = evaluate_model(_dacm_model_path, global_test_loaders)
    print(f"Accuracy : {acc3:.4f}")
    print(f"Precision: {prec3:.4f}")
    print(f"Recall   : {rec3:.4f}  <-- DACM VERIFICATION GATE")
    print(f"F1 Score : {f13:.4f}")
    print(f"Confusion Matrix:\n{cm3}")