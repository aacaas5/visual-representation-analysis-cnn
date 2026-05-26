import os
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

epochs = list(range(1, 11))

loss = [
    1.4807,
    1.1271,
    0.9670,
    0.8617,
    0.7743,
    0.7021,
    0.6294,
    0.5662,
    0.5054,
    0.4427
]

training_accuracy = [
    46.87,
    60.19,
    66.06,
    69.60,
    72.90,
    75.48,
    77.93,
    80.29,
    82.15,
    84.45
]


# -------------------------------------------------
# Loss curve
# -------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    loss,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("CNN Training Loss")

plt.grid(True)

loss_path = os.path.join(
    PROJECT_ROOT,
    "figures",
    "training_loss.png"
)

plt.tight_layout()
plt.savefig(loss_path, dpi=300)
plt.close()


# -------------------------------------------------
# Training accuracy
# -------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    training_accuracy,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.title("CNN Training Accuracy")

plt.grid(True)

accuracy_path = os.path.join(
    PROJECT_ROOT,
    "figures",
    "training_accuracy.png"
)

plt.tight_layout()
plt.savefig(accuracy_path, dpi=300)
plt.close()


print("Saved:")
print(loss_path)
print(accuracy_path)

print("\nFinal training accuracy: 84.45%")
print("Final training loss: 0.4427")
print("Test accuracy: 71.62%")
print("Generalization gap: 12.83 percentage points")