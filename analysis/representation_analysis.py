import os
import sys

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.cnn import CNNClassifier


# -------------------------------------------------
# Settings
# -------------------------------------------------

NUM_IMAGES = 3000
BATCH_SIZE = 64

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------------------------------
# Load test dataset
# -------------------------------------------------

transform = transforms.ToTensor()

test_dataset = datasets.CIFAR10(
    root=os.path.join(PROJECT_ROOT, "data"),
    train=False,
    download=True,
    transform=transform
)

subset = Subset(
    test_dataset,
    range(NUM_IMAGES)
)

test_loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -------------------------------------------------
# Load trained CNN
# -------------------------------------------------

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


# -------------------------------------------------
# Extract 128-dimensional learned representations
# -------------------------------------------------

all_features = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        # Convolutional feature extraction
        x = model.features(images)

        # Flatten
        x = torch.flatten(x, 1)

        # First Linear layer
        x = model.classifier[1](x)

        # ReLU
        x = model.classifier[2](x)

        # x now contains 128 learned features
        all_features.append(
            x.cpu()
        )

        all_labels.append(labels)


features = torch.cat(all_features).numpy()
labels = torch.cat(all_labels).numpy()

print("Representation shape:", features.shape)


# -------------------------------------------------
# PCA
# -------------------------------------------------

pca = PCA(n_components=2)

features_2d = pca.fit_transform(features)

print(
    "Explained variance:",
    pca.explained_variance_ratio_
)


# -------------------------------------------------
# Plot representations
# -------------------------------------------------

plt.figure(figsize=(10, 8))

for class_index, class_name in enumerate(classes):

    mask = labels == class_index

    plt.scatter(
        features_2d[mask, 0],
        features_2d[mask, 1],
        s=15,
        alpha=0.6,
        label=class_name
    )


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "CNN Learned Representations of CIFAR-10 Classes"
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()


figure_path = os.path.join(
    PROJECT_ROOT,
    "figures",
    "representation_pca.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nSaved:")
print(figure_path)