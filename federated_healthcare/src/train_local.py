import os
import torch
import torch.nn as nn
from model import ChestCNN
from utils import load_hospital_data

# -------------------------------
# Paths & Config
# -------------------------------
from pathlib import Path
import sys
# Ensure 'src' package is importable regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import resolve_data_path, MODELS_DIR

DATA_PATH = resolve_data_path(os.environ.get("DATA_PATH"), "Hospital_A")
MODEL_PATH = MODELS_DIR / "local_model_hospital_A.pth"

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

print(f"Using device: {device}")

def train_local():
    print("="*60)
    print("Training LOCAL Model exclusively on Hospital A data")
    print("="*60)

    trainloader, testloader = load_hospital_data(DATA_PATH, batch_size=32)

    model = ChestCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            del outputs, loss, images, labels
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(trainloader):.4f}, Accuracy: {accuracy:.2f}%")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print("="*60)
    print(f"Local Model saved to {MODEL_PATH}")
    print("="*60)

if __name__ == "__main__":
    train_local()
