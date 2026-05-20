import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# -------------------------------------------------
# 1. Convert CIFAR-10 images into PyTorch tensors
# -------------------------------------------------

transform = transforms.ToTensor()


# -------------------------------------------------
# 2. Download CIFAR-10 training data
# -------------------------------------------------

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)


# -------------------------------------------------
# 3. Create DataLoader
# -------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# -------------------------------------------------
# 4. CIFAR-10 class names
# -------------------------------------------------

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


# -------------------------------------------------
# 5. Get one batch
# -------------------------------------------------

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("\nFirst 10 labels:")
print(labels[:10])

print("\nFirst 10 class names:")

for label in labels[:10]:
    print(classes[label.item()])


# -------------------------------------------------
# 6. Show some images
# -------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for i, ax in enumerate(axes.flat):

    image = images[i]

    # PyTorch image format:
    # [channels, height, width]
    #
    # Matplotlib expects:
    # [height, width, channels]

    image = image.permute(1, 2, 0)

    ax.imshow(image)
    ax.set_title(classes[labels[i].item()])
    ax.axis("off")


plt.tight_layout()
plt.show()