import torch


# -------------------------------------------------
# 1. Training data
# -------------------------------------------------

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])

# True relationship:
# y = 2x + 1


# -------------------------------------------------
# 2. Parameters
# -------------------------------------------------

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)


# -------------------------------------------------
# 3. Training settings
# -------------------------------------------------

learning_rate = 0.05
epochs = 100


# -------------------------------------------------
# 4. Training loop
# -------------------------------------------------

for epoch in range(epochs):

    # Make predictions
    y_pred = w * x + b

    # Mean squared error
    loss = ((y_pred - y_true) ** 2).mean()

    # Calculate gradients automatically
    loss.backward()

    # Update parameters
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Clear old gradients
    w.grad.zero_()
    b.grad.zero_()

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {loss.item():.6f} | "
            f"w: {w.item():.4f} | "
            f"b: {b.item():.4f}"
        )


print("\nFinal parameters:")
print("w =", w.item())
print("b =", b.item())