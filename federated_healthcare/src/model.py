import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class ChestCNN(nn.Module):
    def __init__(self):
        super(ChestCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)

        self.pool = nn.MaxPool2d(2, 2)

        # For 128x128 input
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train(net, trainloader, epochs, class_weight=1.0):
    # ============================================================
    # DACU DYNAMIC LOSS INJECTION
    # ============================================================
    if class_weight > 1.0:
        weights = torch.tensor([1.0, float(class_weight)], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    net.to(device)
    net.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Memory optimization: free up tensors immediately
            del outputs, loss, images, labels
            
        # Free up PyTorch cache at the end of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_train_loss = (total_loss / total) if total > 0 else 0.0
    train_acc = (correct / total) if total > 0 else 0.0
    return avg_train_loss, train_acc


def train_dp(
    private_net,
    optimizer,
    trainloader,
    privacy_engine,
    epochs=1,
    delta=1e-5,
    class_weight=1.0
):
    # ============================================================
    # DACU DYNAMIC LOSS INJECTION (FOR DP)
    # ============================================================
    if class_weight > 1.0:
        weights = torch.tensor([1.0, float(class_weight)], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    private_net.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = private_net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Memory optimization: free up tensors immediately
            del outputs, loss, images, labels
            
        # Free up PyTorch cache at the end of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    epsilon = privacy_engine.get_epsilon(
        delta=delta
    )

    avg_train_loss = (total_loss / total) if total > 0 else 0.0
    train_acc = (correct / total) if total > 0 else 0.0
    return epsilon, avg_train_loss, train_acc


def test(net, testloader):
    criterion = nn.CrossEntropyLoss()

    net.to(device)
    net.eval()

    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0

    return loss / len(testloader), accuracy