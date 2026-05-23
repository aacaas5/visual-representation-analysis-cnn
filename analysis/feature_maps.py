import os
import sys

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.cnn import CNNClassifier


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------------------------------
# Load one CIFAR-10 test image
# -------------------------------------------------

transform = transforms.ToTensor()

test_dataset = datasets.CIFAR10(
    root=os.path.join(PROJECT_ROOT, "data"),
    train=False,
    download=True,
    transform=transform
)

image, label = test_dataset[0]

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

print("True class:", classes[label])


# -------------------------------------------------
# Load trained model
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
# Prepare image
# -------------------------------------------------

input_image = image.unsqueeze(0).to(device)

print("Input shape:", input_image.shape)


# -------------------------------------------------
# Pass image through CNN manually
# -------------------------------------------------

conv1 = model.features[0]
relu1 = model.features[1]
pool1 = model.features[2]

conv2 = model.features[3]
relu2 = model.features[4]
pool2 = model.features[5]


x = conv1(input_image)
x = relu1(x)

first_layer_features = x.detach().cpu()

x = pool1(x)

x = conv2(x)
x = relu2(x)

second_layer_features = x.detach().cpu()


print("First feature map shape:", first_layer_features.shape)
print("Second feature map shape:", second_layer_features.shape)


# -------------------------------------------------
# Show original image
# -------------------------------------------------

plt.figure(figsize=(4, 4))

plt.imshow(
    image.permute(1, 2, 0)
)

plt.title(
    f"Original Image - {classes[label]}"
)

plt.axis("off")
plt.tight_layout()

original_path = os.path.join(
    PROJECT_ROOT,
    "figures",
    "feature_original.png"
)

plt.savefig(
    original_path,
    dpi=300
)

plt.show()


# -------------------------------------------------
# Visualize first convolution features
# -------------------------------------------------

fig, axes = plt.subplots(
    4,
    4,
    figsize=(8, 8)
)

for i, ax in enumerate(axes.flat):

    feature_map = first_layer_features[0, i]

    ax.imshow(
        feature_map,
        cmap="gray"
    )

    ax.set_title(
        f"Feature {i + 1}"
    )

    ax.axis("off")


plt.suptitle(
    "First Convolutional Layer Feature Maps"
)

plt.tight_layout()

first_path = os.path.join(
    PROJECT_ROOT,
    "figures",
    "first_layer_feature_maps.png"
)

plt.savefig(
    first_path,
    dpi=300
)

plt.show()


# -------------------------------------------------
# Visualize second convolution features
# -------------------------------------------------

fig, axes = plt.subplots(
    4,
    4,
    figsize=(8, 8)
)

for i, ax in enumerate(axes.flat):

    feature_map = second_layer_features[0, i]

    ax.imshow(
        feature_map,
        cmap="gray"
    )

    ax.set_title(
        f"Feature {i + 1}"
    )

    ax.axis("off")


plt.suptitle(
    "Second Convolutional Layer Feature Maps"
)

plt.tight_layout()

second_path = os.path.join(
    PROJECT_ROOT,
    "figures",
    "second_layer_feature_maps.png"
)

plt.savefig(
    second_path,
    dpi=300
)

plt.show()


print("\nSaved:")
print(original_path)
print(first_path)
print(second_path)