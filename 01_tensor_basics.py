import torch


# -------------------------------------------------
# 1. Creating tensors
# -------------------------------------------------

scalar = torch.tensor(5.0)

vector = torch.tensor([1.0, 2.0, 3.0])

matrix = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

print("Scalar:")
print(scalar)
print("Shape:", scalar.shape)

print("\nVector:")
print(vector)
print("Shape:", vector.shape)

print("\nMatrix:")
print(matrix)
print("Shape:", matrix.shape)


# -------------------------------------------------
# 2. Tensor dimensions
# -------------------------------------------------

print("\nNumber of dimensions:")

print("Scalar dimensions:", scalar.ndim)
print("Vector dimensions:", vector.ndim)
print("Matrix dimensions:", matrix.ndim)


# -------------------------------------------------
# 3. Tensor data types
# -------------------------------------------------

print("\nData types:")

print("Scalar dtype:", scalar.dtype)
print("Vector dtype:", vector.dtype)
print("Matrix dtype:", matrix.dtype)


# -------------------------------------------------
# 4. Basic tensor operations
# -------------------------------------------------

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("\nAddition:")
print(a + b)

print("\nSubtraction:")
print(a - b)

print("\nElement-wise multiplication:")
print(a * b)

print("\nElement-wise division:")
print(a / b)

print("\nDot product:")
print(torch.dot(a, b))


# -------------------------------------------------
# 5. Random tensors
# -------------------------------------------------

random_tensor = torch.rand(3, 4)

print("\nRandom tensor:")
print(random_tensor)

print("Random tensor shape:")
print(random_tensor.shape)


# -------------------------------------------------
# 6. Zeros and ones
# -------------------------------------------------

zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)

print("\nZeros:")
print(zeros)

print("\nOnes:")
print(ones)


# -------------------------------------------------
# 7. Reshaping tensors
# -------------------------------------------------

numbers = torch.tensor([
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
])

reshaped = numbers.reshape(2, 3)

print("\nOriginal tensor:")
print(numbers)

print("\nReshaped tensor:")
print(reshaped)

print("Reshaped shape:")
print(reshaped.shape)


# -------------------------------------------------
# 8. Image-like tensor
# -------------------------------------------------

image = torch.rand(3, 32, 32)

print("\nSingle RGB image tensor shape:")
print(image.shape)


# -------------------------------------------------
# 9. Batch of images
# -------------------------------------------------

batch = torch.rand(64, 3, 32, 32)

print("\nBatch shape:")
print(batch.shape)


# -------------------------------------------------
# 10. CPU / GPU device
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nAvailable device:")
print(device)

batch = batch.to(device)

print("\nBatch device:")
print(batch.device)