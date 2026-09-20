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
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # For 128x128 input
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(net, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    net.to(device)
    net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Memory optimization: free up tensors immediately
            del outputs, loss, images, labels
            
        # Free up PyTorch cache at the end of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def train_dp(
    private_net,
    optimizer,
    trainloader,
    privacy_engine,
    epochs=1,
    delta=1e-5
):
    criterion = nn.CrossEntropyLoss()
    private_net.train()

    for epoch in range(epochs):

        for images, labels in trainloader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = private_net(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()
            
            # Memory optimization: free up tensors immediately
            del outputs, loss, images, labels
            
        # Free up PyTorch cache at the end of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    epsilon = privacy_engine.get_epsilon(
        delta=delta
    )

    return epsilon

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