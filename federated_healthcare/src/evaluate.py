import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# Load environment variables from .env file if present
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                k = key.strip()
                if k not in os.environ:
                    os.environ[k] = val.strip()


from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from sklearn.preprocessing import label_binarize

from model import ChestCNN
from utils import load_hospital_data

# -------------------------------
# Device
# -------------------------------

FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
if FORCE_CPU:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# -------------------------------
# Paths & Config
# -------------------------------

USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "1") == "1"
USE_DP = os.environ.get("USE_DP", "0") == "1"
if USE_DP:
    SUFFIX = "c_dp"
elif USE_QUANTIZATION:
    SUFFIX = "b_quantized"
else:
    SUFFIX = "a_pure"

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "models",
    f"global_model_{SUFFIX}.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dashboard",
    "plots",
    SUFFIX
)

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "dashboard",
    "classification_reports"
)
os.makedirs(REPORTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(
    REPORTS_DIR,
    f"classification_report_{SUFFIX}.md"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Load Model
# -------------------------------

model = ChestCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Global model loaded successfully.")

# -------------------------------
# Load Test Data
# -------------------------------

# DATA_PATH = os.path.join(
#     os.path.dirname(BASE_DIR),
#     "data",
#     "hospital_A"
# )
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "hospital_A"
)
_, testloader = load_hospital_data(DATA_PATH)

print("Test dataset loaded.")

# -------------------------------
# Prediction
# -------------------------------

all_labels = []
all_predictions = []
all_probabilities = []
all_images = []

with torch.no_grad():

    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)

        all_images.extend(images.cpu())

        all_labels.extend(labels.cpu().numpy())

        all_predictions.extend(predicted.cpu().numpy())

        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )

all_labels = np.array(all_labels)

all_predictions = np.array(all_predictions)

all_probabilities = np.array(all_probabilities)

print("Prediction completed.")

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")
plt.colorbar()

classes = ["NORMAL", "PNEUMONIA"]

plt.xticks(np.arange(2), classes)
plt.yticks(np.arange(2), classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=12,
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

print("Confusion Matrix saved.")

# -------------------------------
# ROC Curve
# -------------------------------

fpr, tpr, _ = roc_curve(
    all_labels,
    all_probabilities
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    ),
    dpi=300
)

plt.close()

print("ROC Curve saved.")

# -------------------------------
# Precision Recall Curve
# -------------------------------

precision, recall, _ = precision_recall_curve(
    all_labels,
    all_probabilities
)

ap = average_precision_score(
    all_labels,
    all_probabilities
)

plt.figure(figsize=(7, 6))

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f"AP = {ap:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")

plt.grid(True)

plt.legend()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "pr_curve.png"
    ),
    dpi=300
)

plt.close()

print("Precision Recall Curve saved.")

# -------------------------------
# Sample Predictions
# -------------------------------

plt.figure(figsize=(12, 12))

indices = random.sample(
    range(len(all_images)),
    min(9, len(all_images))
)

for i, idx in enumerate(indices):

    image = all_images[idx].squeeze().numpy()

    actual = classes[all_labels[idx]]

    predicted = classes[all_predictions[idx]]

    plt.subplot(3, 3, i + 1)

    plt.imshow(image, cmap="gray")

    plt.title(
        f"Actual : {actual}\nPred : {predicted}",
        fontsize=9
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "predictions.png"
    ),
    dpi=300
)

plt.close()

print("Sample Predictions saved.")

# -------------------------------
# Classification Report
# -------------------------------

report = classification_report(
    all_labels,
    all_predictions,
    target_names=classes,
    digits=4,
)

with open(REPORT_PATH, "w") as f:
    f.write("# Chest X-ray Pneumonia Classification Report\n\n")
    f.write("```text\n")
    f.write(report)
    f.write("\n```\n")

print("\nClassification Report")
print("=" * 60)
print(report)

print(f"\nClassification report saved to:\n{REPORT_PATH}")

# -------------------------------
# Overall Metrics
# -------------------------------

cm = confusion_matrix(all_labels, all_predictions)

TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

accuracy = (TP + TN) / cm.sum()

precision = TP / (TP + FP) if (TP + FP) > 0 else 0

recall = TP / (TP + FN) if (TP + FN) > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)

print("\n")
print("=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print(f"ROC AUC  : {roc_auc:.4f}")
print(f"PR AUC   : {ap:.4f}")

print("\nGenerated Files")
print("-" * 60)

print(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
print(os.path.join(OUTPUT_DIR, "roc_curve.png"))
print(os.path.join(OUTPUT_DIR, "pr_curve.png"))
print(os.path.join(OUTPUT_DIR, "predictions.png"))
print(REPORT_PATH)

print("\nEvaluation Completed Successfully.")
print("=" * 60)