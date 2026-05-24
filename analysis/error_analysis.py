import os
import sys

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.cnn import CNNClassifier


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
# Test data
# -------------------------------------------------

transform = transforms.ToTensor()

test_dataset = datasets.CIFAR10(
    root=os.path.join(PROJECT_ROOT, "data"),
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


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
# Collect predictions
# -------------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())


# -------------------------------------------------
# Confusion matrix
# -------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

fig, ax = plt.subplots(figsize=(10, 10))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    colorbar=False
)

plt.title("CIFAR-10 Confusion Matrix")
plt.tight_layout()

figure_path = os.path.join(
    PROJECT_ROOT,
    "figures",
    "confusion_matrix.png"
)

plt.savefig(
    figure_path,
    dpi=300
)

plt.show()


# -------------------------------------------------
# Per-class accuracy
# -------------------------------------------------

print("\nPer-class accuracy:\n")

for i, class_name in enumerate(classes):

    correct = cm[i, i]
    total = cm[i].sum()

    accuracy = 100 * correct / total

    print(
        f"{class_name:10s}: "
        f"{accuracy:.2f}%"
    )


print("\nConfusion matrix saved to:")
print(figure_path)

# -------------------------------------------------
# Confidence analysis: correct vs incorrect
# -------------------------------------------------

correct_confidences = []
wrong_confidences = []

correct_examples = []
wrong_examples = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predicted = torch.max(
            probabilities,
            dim=1
        )

        for i in range(len(labels)):

            confidence = confidences[i].item()
            prediction = predicted[i].item()
            true_label = labels[i].item()

            if prediction == true_label:

                correct_confidences.append(
                    confidence
                )

                if len(correct_examples) < 8:
                    correct_examples.append(
                        (
                            images[i].cpu(),
                            true_label,
                            prediction,
                            confidence
                        )
                    )

            else:

                wrong_confidences.append(
                    confidence
                )

                if len(wrong_examples) < 8:
                    wrong_examples.append(
                        (
                            images[i].cpu(),
                            true_label,
                            prediction,
                            confidence
                        )
                    )


# -------------------------------------------------
# Average confidence
# -------------------------------------------------

average_correct_confidence = (
    sum(correct_confidences)
    / len(correct_confidences)
)

average_wrong_confidence = (
    sum(wrong_confidences)
    / len(wrong_confidences)
)

print("\nConfidence analysis:")

print(
    f"Average confidence when correct: "
    f"{average_correct_confidence * 100:.2f}%"
)

print(
    f"Average confidence when wrong: "
    f"{average_wrong_confidence * 100:.2f}%"
)


# -------------------------------------------------
# Show wrong predictions
# -------------------------------------------------

fig, axes = plt.subplots(
    2,
    4,
    figsize=(12, 6)
)

for i, ax in enumerate(axes.flat):

    image, true_label, prediction, confidence = wrong_examples[i]

    image = image.permute(
        1,
        2,
        0
    )

    ax.imshow(image)

    ax.set_title(
        f"True: {classes[true_label]}\n"
        f"Pred: {classes[prediction]}\n"
        f"Conf: {confidence * 100:.1f}%"
    )

    ax.axis("off")


plt.suptitle(
    "Incorrect Predictions and Model Confidence"
)

plt.tight_layout()

wrong_path = os.path.join(
    PROJECT_ROOT,
    "figures",
    "wrong_predictions.png"
)

plt.savefig(
    wrong_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nSaved:")
print(wrong_path)