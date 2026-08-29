import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ChestCNN
from utils import load_hospital_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# -------------------------------
# Paths & Config
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "1") == "1"
USE_DP = os.environ.get("USE_DP", "0") == "1"
if USE_DP:
    SUFFIX = "c_dp"
elif USE_QUANTIZATION:
    SUFFIX = "b_quantized"
else:
    SUFFIX = "a_pure"

LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "federated_healthcare", "models", "local_model_hospital_A.pth")
GLOBAL_MODEL_PATH = os.path.join(BASE_DIR, "federated_healthcare", "models", f"global_model_{SUFFIX}.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "federated_healthcare", "dashboard", "plots", "comparisons")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Device
# -------------------------------
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
if FORCE_CPU:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

def evaluate_model(model, testloader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())
            
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds) * 100,
        "precision": precision_score(all_labels, all_preds, zero_division=0) * 100,
        "recall": recall_score(all_labels, all_preds, zero_division=0) * 100,
        "f1": f1_score(all_labels, all_preds, zero_division=0) * 100,
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs)
    }
    return metrics

def run_comparison():
    print("="*60)
    print("Running Non-IID vs Centralized Federated Model Study")
    print("="*60)

    # 1. Load Models
    local_model = ChestCNN().to(device)
    global_model = ChestCNN().to(device)

    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"Error: Local model not found at {LOCAL_MODEL_PATH}. Run train_local.py first.")
        return
    if not os.path.exists(GLOBAL_MODEL_PATH):
        print(f"Error: Global model not found at {GLOBAL_MODEL_PATH}. Run FL server first.")
        return

    local_model.load_state_dict(torch.load(LOCAL_MODEL_PATH, map_location=device))
    global_model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location=device))
    print("Models loaded successfully.")

    # 2. Load Datasets
    hospitals = ["hospital_A", "hospital_B", "hospital_C"]
    
    local_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    global_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    
    hosp_b_data = {}

    for hosp in hospitals:
        data_path = os.path.join(BASE_DIR, "data", hosp)
        _, testloader = load_hospital_data(data_path, batch_size=32)
        
        print(f"\nEvaluating on {hosp} test set...")
        loc_res = evaluate_model(local_model, testloader)
        glob_res = evaluate_model(global_model, testloader)
        
        if hosp == "hospital_B":
            hosp_b_data["local"] = loc_res
            hosp_b_data["global"] = glob_res
        
        print(f"Local Model Accuracy: {loc_res['accuracy']:.2f}% | F1: {loc_res['f1']:.2f}%")
        print(f"Global FL Model Accuracy: {glob_res['accuracy']:.2f}% | F1: {glob_res['f1']:.2f}%")
        
        for metric in ["accuracy", "precision", "recall", "f1"]:
            local_metrics[metric].append(loc_res[metric])
            global_metrics[metric].append(glob_res[metric])

    # 3. Generate Side-by-Side Accuracy Plot
    x = np.arange(len(hospitals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, local_metrics["accuracy"], width, label='Local Model (Trained only on A)', color='#ff7f0e')
    rects2 = ax.bar(x + width/2, global_metrics["accuracy"], width, label='Global FL Model', color='#1f77b4')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Non-IID Study: Local vs. Federated Learning Performance', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Hospital A\n(Skewed Normal)', 'Hospital B\n(Skewed Pneumonia)', 'Hospital C\n(Balanced)'], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plot_path_acc = os.path.join(OUTPUT_DIR, "non_iid_study_accuracy.png")
    plt.savefig(plot_path_acc, dpi=300)
    plt.close()
    
    # 4. Generate F1 Score Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, local_metrics["f1"], width, label='Local Model (Trained only on A)', color='#d62728')
    rects2 = ax.bar(x + width/2, global_metrics["f1"], width, label='Global FL Model', color='#2ca02c')

    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('Robustness Test: F1 Score Comparison Across Demographics', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Hospital A\n(Skewed Normal)', 'Hospital B\n(Skewed Pneumonia)', 'Hospital C\n(Balanced)'], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plot_path_f1 = os.path.join(OUTPUT_DIR, "non_iid_study_f1.png")
    plt.savefig(plot_path_f1, dpi=300)
    plt.close()

    # 5. Side-by-Side Confusion Matrix on Hospital B
    loc_cm = confusion_matrix(hosp_b_data["local"]["labels"], hosp_b_data["local"]["preds"])
    glob_cm = confusion_matrix(hosp_b_data["global"]["labels"], hosp_b_data["global"]["preds"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    classes = ["NORMAL", "PNEUMONIA"]
    
    for idx, (ax, cm, title) in enumerate(zip(axes, [loc_cm, glob_cm], ["Local Model on Hosp B", "Federated Model on Hosp B"])):
        im = ax.imshow(cm, cmap="Blues" if idx == 1 else "Oranges")
        ax.set_title(title, fontsize=14)
        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i,j] < cm.max()/2 else "white", fontsize=12)
                
    plt.tight_layout()
    plot_path_cm = os.path.join(OUTPUT_DIR, "non_iid_study_cm_hosp_B.png")
    plt.savefig(plot_path_cm, dpi=300)
    plt.close()

    # 6. ROC Curve for Hospital B
    fpr_loc, tpr_loc, _ = roc_curve(hosp_b_data["local"]["labels"], hosp_b_data["local"]["probs"])
    fpr_glob, tpr_glob, _ = roc_curve(hosp_b_data["global"]["labels"], hosp_b_data["global"]["probs"])
    
    auc_loc = auc(fpr_loc, tpr_loc)
    auc_glob = auc(fpr_glob, tpr_glob)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_loc, tpr_loc, linewidth=2, color='#ff7f0e', label=f"Local Model (AUC = {auc_loc:.4f})")
    plt.plot(fpr_glob, tpr_glob, linewidth=2, color='#1f77b4', label=f"Federated Model (AUC = {auc_glob:.4f})")
    plt.plot([0, 1], [0, 1], "--", color='gray')
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve on Hospital B (Skewed Pneumonia)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plot_path_roc = os.path.join(OUTPUT_DIR, "non_iid_study_roc_hosp_B.png")
    plt.savefig(plot_path_roc, dpi=300)
    plt.close()

    print("\n"+"="*60)
    print("Comparison study completed! Plots generated:")
    print(f"- {plot_path_acc}")
    print(f"- {plot_path_f1}")
    print(f"- {plot_path_cm}")
    print(f"- {plot_path_roc}")
    print("="*60)

if __name__ == "__main__":
    run_comparison()
