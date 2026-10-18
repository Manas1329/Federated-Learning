# import os
# import torch
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms
# from PIL import Image

# def load_hospital_data(data_path, batch_size=32, img_size=128):
#     """
#     Load chest X-ray images from hospital-specific folder structure.
    
#     Expected folder structure:
#         data_path/
#             train/
#                 NORMAL/
#                 PNEUMONIA/
#             test/
#                 NORMAL/
#                 PNEUMONIA/
    
#     Returns:
#         trainloader, testloader (DataLoader objects)
#     """
#     transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#     ])

    # train_path = os.path.join(data_path, "train")
    # test_path = os.path.join(data_path, "test")

    # # If actual data folders don't exist, create dummy data for testing
    # if not os.path.exists(train_path):
    #     os.makedirs(os.path.join(train_path, "NORMAL"), exist_ok=True)
    #     os.makedirs(os.path.join(train_path, "PNEUMONIA"), exist_ok=True)
    #     # Create a few dummy black images so DataLoader can initialize
    #     for i in range(4):
    #         dummy_img = Image.new("L", (img_size, img_size), 0)
    #         dummy_img.save(os.path.join(train_path, "NORMAL", f"dummy_{i}.png"))
    #         dummy_img.save(os.path.join(train_path, "PNEUMONIA", f"dummy_{i}.png"))

    # if not os.path.exists(test_path):
    #     os.makedirs(os.path.join(test_path, "NORMAL"), exist_ok=True)
    #     os.makedirs(os.path.join(test_path, "PNEUMONIA"), exist_ok=True)
    #     for i in range(2):
    #         dummy_img = Image.new("L", (img_size, img_size), 0)
    #         dummy_img.save(os.path.join(test_path, "NORMAL", f"dummy_{i}.png"))
    #         dummy_img.save(os.path.join(test_path, "PNEUMONIA", f"dummy_{i}.png"))

    # train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    # test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    # trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print(f"Loaded {len(train_dataset)} training and {len(test_dataset)} test samples from {data_path}")
    
    # return trainloader, testloader
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from pathlib import Path

class TransformedSubset(torch.utils.data.Dataset):
    """
    Dataset wrapper to apply split-specific transforms to a PyTorch Subset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def load_hospital_data(data_path, batch_size=32, img_size=128):
    """
    Load chest X-ray images from hospital-specific folder structure with medical-safe augmentations.

    Expected folder structure:
        data_path/
            train/
                NORMAL/
                PNEUMONIA/

    Returns:
        trainloader, testloader (DataLoader objects)
    """

    train_path = Path(data_path) / "train"

    # Load raw dataset without transforms to apply split-specific pipelines
    full_dataset = datasets.ImageFolder(root=train_path)

    # Split into 80% training and 20% testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_subset, test_subset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Medical-safe training data augmentation pipeline
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    # Unaugmented test evaluation pipeline
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    test_dataset = TransformedSubset(test_subset, transform=test_transform)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Loaded {len(train_dataset)} training samples (augmented)")
    print(f"Loaded {len(test_dataset)} testing samples (unaugmented)")

    return trainloader, testloader


def load_partitions(client_id="Hospital_A", batch_size=32):
    """
    Load data partitions for a hospital client.
    Returns: trainloader, testloader, num_examples
    """
    from paths import resolve_data_path
    data_path = resolve_data_path(None, client_id)
    trainloader, testloader = load_hospital_data(data_path, batch_size=batch_size)
    num_examples = {
        "trainset": len(trainloader.dataset),
        "testset": len(testloader.dataset)
    }
    return trainloader, testloader, num_examples
