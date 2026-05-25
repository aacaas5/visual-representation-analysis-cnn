import os
import sys

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms


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
# Load CIFAR-10 test image
# -------------------------------------------------

transform = transforms.ToTensor()

test_dataset = datasets.CIFAR10(
    root=os.path.join(PROJECT_ROOT, "data"),
    train=False,
    download=True,
    transform=transform
)

image, true_label = test_dataset[0]

input_image = image.unsqueeze(0).to(device)


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
# Capture activations and gradients
# -------------------------------------------------

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


target_layer = model.features[3]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# -------------------------------------------------
# Forward pass
# -------------------------------------------------

output = model(input_image)

predicted_class = output.argmax(dim=1).item()

print("True class:", classes[true_label])
print("Predicted class:", classes[predicted_class])


# -------------------------------------------------
# Backward pass for predicted class
# -------------------------------------------------

model.zero_grad()

score = output[0, predicted_class]

score.backward()


# -------------------------------------------------
# Create Grad-CAM
# -------------------------------------------------

weights = gradients.mean(
    dim=(2, 3),
    keepdim=True
)

cam = (
    weights * activations
).sum(dim=1, keepdim=True)

cam = F.relu(cam)

cam = F.interpolate(
    cam,
    size=(32, 32),
    mode="bilinear",
    align_corners=False
)

cam = cam.squeeze().detach().cpu().numpy()


# Normalize heatmap
cam -= cam.min()

if cam.max() > 0:
    cam /= cam.max()


# -------------------------------------------------
# Prepare original image
# -------------------------------------------------

original_image = image.permute(
    1,
    2,
    0
).numpy()


# -------------------------------------------------
# Display result
# -------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 4)
)


axes[0].imshow(original_image)
axes[0].set_title(
    f"Original\nTrue: {classes[true_label]}"
)
axes[0].axis("off")


axes[1].imshow(cam, cmap="jet")
axes[1].set_title("Grad-CAM Heatmap")
axes[1].axis("off")


axes[2].imshow(original_image)

axes[2].imshow(
    cam,
    cmap="jet",
    alpha=0.45
)

axes[2].set_title(
    f"Attention Overlay\nPredicted: {classes[predicted_class]}"
)

axes[2].axis("off")


plt.tight_layout()


figure_path = os.path.join(
    PROJECT_ROOT,
    "figures",
    "gradcam_example.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nSaved:")
print(figure_path)

# -------------------------------------------------
# Compare one correct and one wrong prediction
# -------------------------------------------------

def make_gradcam_for_image(image, true_label, save_name):

    global activations, gradients

    input_image = image.unsqueeze(0).to(device)

    output = model(input_image)

    predicted_class = output.argmax(dim=1).item()

    probability = torch.softmax(output, dim=1)

    confidence = probability[0, predicted_class].item()

    model.zero_grad()

    score = output[0, predicted_class]
    score.backward()

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activations
    ).sum(dim=1, keepdim=True)

    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=(32, 32),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().detach().cpu().numpy()

    cam -= cam.min()

    if cam.max() > 0:
        cam /= cam.max()

    original_image = image.permute(
        1,
        2,
        0
    ).numpy()

    return (
        original_image,
        cam,
        true_label,
        predicted_class,
        confidence
    )


# -------------------------------------------------
# Find one correct and one wrong example
# -------------------------------------------------

correct_example = None
wrong_example = None

for i in range(len(test_dataset)):

    image_i, label_i = test_dataset[i]

    input_i = image_i.unsqueeze(0).to(device)

    with torch.no_grad():
        output_i = model(input_i)

    prediction_i = output_i.argmax(dim=1).item()

    if prediction_i == label_i and correct_example is None:
        correct_example = (image_i, label_i)

    if prediction_i != label_i and wrong_example is None:
        wrong_example = (image_i, label_i)

    if correct_example is not None and wrong_example is not None:
        break


# -------------------------------------------------
# Generate both Grad-CAM results
# -------------------------------------------------

correct_result = make_gradcam_for_image(
    correct_example[0],
    correct_example[1],
    "correct"
)

wrong_result = make_gradcam_for_image(
    wrong_example[0],
    wrong_example[1],
    "wrong"
)


# -------------------------------------------------
# Plot comparison
# -------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)


results = [
    ("Correct Prediction", correct_result),
    ("Wrong Prediction", wrong_result)
]


for row, (title, result) in enumerate(results):

    original_image, cam, true_label, predicted_class, confidence = result

    axes[row, 0].imshow(original_image)

    axes[row, 0].set_title(
        f"{title}\n"
        f"True: {classes[true_label]}"
    )

    axes[row, 0].axis("off")


    axes[row, 1].imshow(
        cam,
        cmap="jet"
    )

    axes[row, 1].set_title(
        "Grad-CAM"
    )

    axes[row, 1].axis("off")


    axes[row, 2].imshow(
        original_image
    )

    axes[row, 2].imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    axes[row, 2].set_title(
        f"Pred: {classes[predicted_class]}\n"
        f"Confidence: {confidence * 100:.1f}%"
    )

    axes[row, 2].axis("off")


plt.suptitle(
    "Grad-CAM Comparison: Correct vs Incorrect Prediction"
)

plt.tight_layout()


comparison_path = os.path.join(
    PROJECT_ROOT,
    "figures",
    "gradcam_correct_vs_wrong.png"
)

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nSaved comparison:")
print(comparison_path)