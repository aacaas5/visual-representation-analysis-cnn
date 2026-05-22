import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.cnn import CNNClassifier


BATCH_SIZE = 64

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Test data
transform = transforms.ToTensor()

test_dataset = datasets.CIFAR10(
    root=os.path.join(PROJECT_ROOT, "data"),
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Load trained CNN
model = CNNClassifier().to(device)

checkpoint_path = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "cnn_cifar10.pth"
)

model.load_state_dict(
    torch.load(checkpoint_path, map_location=device)
)

model.eval()


# Evaluation
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total

print("\nTest images:", total)
print("Correct predictions:", correct)
print(f"Test accuracy: {accuracy:.2f}%")