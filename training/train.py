import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Allow imports from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.cnn import CNNClassifier


# -------------------------------------------------
# Settings
# -------------------------------------------------

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------------------------------
# Data
# -------------------------------------------------

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root=os.path.join(PROJECT_ROOT, "data"),
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# -------------------------------------------------
# Model
# -------------------------------------------------

model = CNNClassifier().to(device)


# -------------------------------------------------
# Loss function
# -------------------------------------------------

criterion = nn.CrossEntropyLoss()


# -------------------------------------------------
# Optimizer
# -------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------------------------------
# Training
# -------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    average_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )


# -------------------------------------------------
# Save trained model
# -------------------------------------------------

checkpoint_path = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "cnn_cifar10.pth"
)

torch.save(
    model.state_dict(),
    checkpoint_path
)

print("\nTraining complete.")
print("Model saved to:", checkpoint_path)