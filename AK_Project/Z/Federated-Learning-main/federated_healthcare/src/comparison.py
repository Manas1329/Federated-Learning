import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import ChestCNN
from src.utils import load_hospital_data
from src.paths import DATA_DIR, MODELS_DIR, DASHBOARD_DIR, RESULTS_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, testloader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    if len(np.unique(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
    else:
        roc_auc = float('nan')
        pr_auc = float('nan')
        
    return acc, prec, rec, f1, roc_auc, pr_auc

def main():
    model_a_path = MODELS_DIR / "local_model_hospital_A.pth"
    global_model_path = MODELS_DIR / "global_model_a_pure.pth"
    
    if not model_a_path.exists():
        print(f"Error: Could not find {model_a_path}")
        return
    if not global_model_path.exists():
        print(f"Error: Could not find {global_model_path}")
        return
        
    print(f"Loading A_only model from {model_a_path}")
    model_a = ChestCNN().to(device)
    model_a.load_state_dict(torch.load(model_a_path, map_location=device))
    
    print(f"Loading Federated model from {global_model_path}")
    model_global = ChestCNN().to(device)
    model_global.load_state_dict(torch.load(global_model_path, map_location=device))
    
    hospitals = ["Hospital_A", "Hospital_B", "Hospital_C"]
    
    results = []
    
    print("\n--- Starting Evaluation ---")
    for hospital in hospitals:
        data_path = DATA_DIR / hospital.lower()
        if not data_path.exists():
            print(f"Warning: Data path {data_path} does not exist, skipping {hospital}.")
            continue
            
        print(f"\nEvaluating on {hospital} test data...")
        _, testloader = load_hospital_data(str(data_path))
        
        # Eval A_only
        acc_a, prec_a, rec_a, f1_a, roc_a, pr_a = evaluate_model(model_a, testloader)
        results.append(["A_only", hospital, acc_a, prec_a, rec_a, f1_a, roc_a, pr_a])
        
        # Eval Federated
        acc_g, prec_g, rec_g, f1_g, roc_g, pr_g = evaluate_model(model_global, testloader)
        results.append(["Federated", hospital, acc_g, prec_g, rec_g, f1_g, roc_g, pr_g])
        
    df = pd.DataFrame(results, columns=["model", "hospital", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
    results_path = RESULTS_DIR / "comparison_results.csv"
    df.to_csv(results_path, index=False)
    
    print("\n--- Comparison Results ---")
    print(df.to_string(index=False))
    
    # Calculate generalization gap for A_only
    a_only_df = df[df["model"] == "A_only"]
    if len(a_only_df) == 3:
        acc_A = a_only_df[a_only_df["hospital"] == "Hospital_A"]["accuracy"].values[0]
        acc_B = a_only_df[a_only_df["hospital"] == "Hospital_B"]["accuracy"].values[0]
        acc_C = a_only_df[a_only_df["hospital"] == "Hospital_C"]["accuracy"].values[0]
        gap_a = acc_A - ((acc_B + acc_C) / 2)
        print(f"\nA_only Generalization Gap (Acc_A - Avg(Acc_B, Acc_C)): {gap_a:.4f}")
        
    # Calculate generalization gap for Federated
    fed_df = df[df["model"] == "Federated"]
    if len(fed_df) == 3:
        acc_A = fed_df[fed_df["hospital"] == "Hospital_A"]["accuracy"].values[0]
        acc_B = fed_df[fed_df["hospital"] == "Hospital_B"]["accuracy"].values[0]
        acc_C = fed_df[fed_df["hospital"] == "Hospital_C"]["accuracy"].values[0]
        gap_f = acc_A - ((acc_B + acc_C) / 2)
        print(f"Federated Generalization Gap (Acc_A - Avg(Acc_B, Acc_C)): {gap_f:.4f}")
        
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
