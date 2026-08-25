# Visual Representation Analysis in Convolutional Neural Networks

### Understanding How Convolutional Neural Networks Learn, Transform, and Use Visual Features

This project studies what happens **inside a Convolutional Neural Network (CNN)** while it learns to classify natural images.

Instead of focusing only on final classification accuracy, the project investigates:

- how visual features evolve through convolutional layers,
- how CIFAR-10 classes are organized in learned representation space,
- which classes are confused,
- whether incorrect predictions can still have high confidence,
- and which image regions influence CNN decisions using Grad-CAM.

The main goal is to move beyond:

> "How accurate is the model?"

toward:

> "What visual representations did the model learn, and how do those representations influence its decisions?"

---

## Dataset

The project uses the **CIFAR-10 dataset**, containing 10 natural-image categories:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Each image has size:

```text
3 × 32 × 32
```

which corresponds to:

```text
RGB channels × height × width
```

Dataset sizes:

```text
Training images: 50,000
Test images:     10,000
```

---

## CNN Architecture

The model uses a compact CNN architecture:

```text
Input
[3 × 32 × 32]

      ↓

Conv2D
3 → 32 channels
3 × 3 kernel

      ↓

ReLU

      ↓

MaxPool
32 × 32 → 16 × 16

      ↓

Conv2D
32 → 64 channels
3 × 3 kernel

      ↓

ReLU

      ↓

MaxPool
16 × 16 → 8 × 8

      ↓

Flatten

64 × 8 × 8
= 4096 features

      ↓

Linear
4096 → 128

      ↓

ReLU

      ↓

Linear
128 → 10 class logits
```

The convolutional layers learn spatial visual patterns, while the fully connected layers combine the learned features for classification.

---

## Training Configuration

The model was trained using:

```text
Optimizer:     Adam
Learning rate: 0.001
Batch size:    64
Epochs:        10
Loss:          Cross-Entropy Loss
```

GPU acceleration with PyTorch CUDA was used when available.

---

## Training Results

| Epoch | Training Loss | Training Accuracy |
|------:|--------------:|------------------:|
| 1 | 1.4807 | 46.87% |
| 2 | 1.1271 | 60.19% |
| 3 | 0.9670 | 66.06% |
| 4 | 0.8617 | 69.60% |
| 5 | 0.7743 | 72.90% |
| 6 | 0.7021 | 75.48% |
| 7 | 0.6294 | 77.93% |
| 8 | 0.5662 | 80.29% |
| 9 | 0.5054 | 82.15% |
| 10 | 0.4427 | 84.45% |

Final training results:

```text
Training Accuracy: 84.45%
Training Loss:     0.4427
Test Accuracy:     71.62%
```

Generalization gap:

```text
84.45% - 71.62%
= 12.83 percentage points
```

### Training Accuracy

![Training Accuracy](figures/training_accuracy.png)

### Training Loss

![Training Loss](figures/training_loss.png)

---

## Test Performance

The trained CNN was evaluated on **10,000 unseen CIFAR-10 test images**.

```text
Correct predictions: 7162
Total test images:   10000
Test accuracy:       71.62%
```

The difference between training and test performance suggests some overfitting and motivates deeper inspection of the learned representations.

---

## Confusion Matrix Analysis

The confusion matrix shows how frequently each class is predicted correctly or confused with another class.

![Confusion Matrix](figures/confusion_matrix.png)

The matrix can be interpreted as:

```text
Rows     = true classes
Columns  = predicted classes
Diagonal = correct predictions
```

### Per-Class Accuracy

| Class | Accuracy |
|---|---:|
| Airplane | 76.6% |
| Automobile | 86.3% |
| Bird | 60.0% |
| Cat | 48.1% |
| Deer | 66.4% |
| Dog | 60.1% |
| Frog | 81.0% |
| Horse | 78.0% |
| Ship | 83.6% |
| Truck | 76.1% |

The strongest performing class was:

```text
Automobile: 86.3%
```

The weakest performing class was:

```text
Cat: 48.1%
```

Several systematic confusions appeared.

Important examples include:

```text
Cat → Dog           167 images
Dog → Cat           159 images
Truck → Automobile  122 images
Cat → Bird           94 images
Bird → Deer          88 images
Deer → Horse         78 images
Airplane → Ship      74 images
```

These results suggest that visually similar categories can develop overlapping internal representations.

---

## Feature Map Analysis

A CNN does not directly transform an image into a class label.

Instead, the image passes through several intermediate feature representations.

The first convolutional layer produces:

```text
32 feature maps
```

The second convolutional layer produces:

```text
64 feature maps
```

Feature maps were extracted from the trained CNN to examine how the input image is transformed internally.

### Original Image

![Original Feature Image](figures/feature_original.png)

### First Convolutional Layer

![First Layer Feature Maps](figures/first_layer_feature_maps.png)

The first convolutional layer preserves more spatial information and responds to relatively simple patterns such as:

- edges,
- color transitions,
- textures,
- local shapes.

### Second Convolutional Layer

![Second Layer Feature Maps](figures/second_layer_feature_maps.png)

The second convolutional layer operates on the features produced by the first layer.

It therefore represents more complex combinations of visual patterns.

The general feature hierarchy can be understood as:

```text
Raw pixels
    ↓
Simple visual patterns
    ↓
Edges and textures
    ↓
More complex feature combinations
    ↓
Learned image representation
```

---

## Learned Representation Space

Before the final classification layer, every image is represented by a:

```text
128-dimensional learned feature vector
```

For example:

```text
Image
   ↓
CNN
   ↓
[128 learned feature values]
```

These 128-dimensional vectors describe how the CNN internally represents each image.

Because 128 dimensions cannot be directly visualized, **Principal Component Analysis (PCA)** was used to reduce the representation space to two dimensions.

```text
128 dimensions
      ↓
     PCA
      ↓
2 dimensions
```

![Representation PCA](figures/representation_pca.png)

Each point represents one CIFAR-10 image.

The point color represents the true image class.

Images with similar learned representations tend to appear closer together in the PCA projection.

Classes with greater overlap can be harder for the classifier to distinguish.

Several animal categories showed substantial overlap in the two-dimensional PCA projection.

This is consistent with the confusion matrix, where classes such as:

```text
cat
dog
bird
deer
```

showed relatively high confusion.

PCA is only a two-dimensional projection of the original 128-dimensional representation space.

Therefore, overlap in the PCA figure does not necessarily mean that the classes are completely inseparable in the full representation space.

---

## PCA Mathematics

For 3000 analyzed images, the representation matrix has shape:

```text
3000 × 128
```

where:

```text
3000 = number of images
128  = learned features per image
```

The feature matrix can be written as:

```text
X ∈ R^(3000 × 128)
```

PCA first centers the data:

```text
X_centered = X - μ
```

where:

```text
μ = mean feature vector
```

The covariance matrix is then conceptually:

```text
C = (1 / (n - 1)) × X_centered^T × X_centered
```

For this project:

```text
X_centered = 3000 × 128
C          = 128 × 128
```

PCA identifies eigenvectors of the covariance matrix corresponding to directions of maximum variance.

The two principal directions form:

```text
W = 128 × 2
```

The projection is then:

```text
Z = X_centered × W
```

where:

```text
X_centered = 3000 × 128
W          = 128 × 2
Z          = 3000 × 2
```

Therefore:

```text
3000 × 128
      ↓
     PCA
      ↓
3000 × 2
```

The number of images remains the same.

Only the number of features used to represent each image is reduced.

Each image therefore becomes:

```text
128 learned values
        ↓
       PCA
        ↓
[x-coordinate, y-coordinate]
        ↓
one point in the visualization
```

---

## Prediction Confidence and Failure Analysis

The CNN produces 10 raw class scores called **logits**:

```text
z_1, z_2, ..., z_10
```

Softmax converts these logits into normalized class probabilities.

For class `i`:

```text
P_i = exp(z_i) / Σ_j exp(z_j)
```

where:

```text
z_i = logit for class i
P_i = softmax probability for class i
```

All probabilities sum to approximately:

```text
Σ_i P_i = 1
```

The class with the largest probability becomes the predicted class.

Incorrect predictions were then inspected individually.

![Incorrect Predictions](figures/wrong_predictions.png)

Several incorrect predictions were made with high confidence.

Examples included:

```text
True: airplane
Predicted: bird
Confidence: 99.9%

True: ship
Predicted: frog
Confidence: 92.7%

True: dog
Predicted: deer
Confidence: 83.5%

True: bird
Predicted: deer
Confidence: 83.0%

True: deer
Predicted: airplane
Confidence: 75.0%

True: dog
Predicted: cat
Confidence: 61.7%
```

These examples demonstrate an important property of neural classifiers:

```text
High confidence ≠ guaranteed correctness
```

A model can produce a dominant class probability even when the prediction is incorrect.

---

## Grad-CAM Visual Explanation

Grad-CAM stands for:

```text
Gradient-weighted Class Activation Mapping
```

It was used to investigate **which image regions contributed most strongly to the CNN's prediction**.

In simple terms, after the CNN makes a prediction, Grad-CAM asks:

> Which parts of the image were most influential for producing this class score?

Grad-CAM uses gradients flowing through a convolutional layer to estimate the importance of each learned feature map.

### Grad-CAM Mathematics

Suppose the convolutional layer produces feature maps:

```text
A_1, A_2, ..., A_k
```

For feature map `A_k`, Grad-CAM computes an importance weight:

```text
alpha_k
=
(1 / (H × W))
×
Σ_i Σ_j
[
∂y_c / ∂A_k(i, j)
]
```

where:

```text
y_c          = score for target class c
A_k          = kth convolutional feature map
H            = feature-map height
W            = feature-map width
∂y_c/∂A_k    = gradient of class score with respect to feature map
```

The gradient answers:

```text
If this feature-map activation changed,
how much would the target class score change?
```

The spatial gradients are averaged to obtain one importance value for each feature map.

The Grad-CAM heatmap is then calculated as:

```text
L_GradCAM^c
=
ReLU(
    Σ_k [ alpha_k × A_k ]
)
```

This means:

```text
Feature map 1 × importance 1
+
Feature map 2 × importance 2
+
Feature map 3 × importance 3
+
...
        ↓
Add them together
        ↓
Apply ReLU
        ↓
Grad-CAM heatmap
```

The ReLU operation is used to keep positive evidence supporting the selected class.

The complete process is:

```text
Input Image
     ↓
CNN Forward Pass
     ↓
Target Class Score y_c
     ↓
Backpropagate Gradient
     ↓
∂y_c / ∂A_k
     ↓
Average Gradients
     ↓
alpha_k
     ↓
Weighted Feature Maps
     ↓
Σ_k alpha_k A_k
     ↓
ReLU
     ↓
Grad-CAM Heatmap
```

### Grad-CAM Example

![Grad-CAM Example](figures/gradcam_example.png)

The heatmap indicates which spatial regions contributed more strongly to the CNN's prediction.

Generally:

```text
Red / Yellow
→ stronger influence on the prediction

Blue
→ weaker influence on the prediction
```

In the example above, the network does not rely exclusively on the central object.

Some highly activated regions also occur in the surrounding image context.

This suggests that the CNN may use a combination of **object features and contextual visual cues** when making its classification decision.

---

## Correct vs Incorrect Grad-CAM

Grad-CAM was also compared between a correctly classified image and an incorrectly classified image.

![Grad-CAM Correct vs Wrong](figures/gradcam_correct_vs_wrong.png)

### Correct Prediction

```text
True class:      cat
Predicted class: cat
Confidence:      51.6%
```

### Incorrect Prediction

```text
True class:      frog
Predicted class: bird
Confidence:      63.6%
```

The comparison shows that an incorrect prediction does not necessarily occur because the CNN completely ignored the object.

Instead, the network may focus on meaningful visual regions but extract features that resemble those associated with another class.

A simplified interpretation is:

```text
Image
  ↓
CNN detects visual patterns
  ↓
Patterns form an internal representation
  ↓
Representation resembles another class
  ↓
Incorrect prediction
```

This connects Grad-CAM with the other findings in the project:

```text
Feature Extraction
        ↓
Learned Representation
        ↓
Representation Overlap
        ↓
Class Confusion
        ↓
Incorrect Prediction
```

Grad-CAM therefore helps explain **where the CNN obtained evidence for a prediction**, while the representation and confusion analyses help explain **why that evidence may lead to the wrong class**.

---

## Core CNN Mathematics

A convolution operation can be understood as a **small trainable filter sliding across an image**.

For an input image `X` and convolution kernel `K`, one output value can be written as:

```text
Y(i, j)
=
Σ_m Σ_n
[
X(i + m, j + n) × K(m, n)
]
+
b
```

where:

```text
X(i + m, j + n) = input pixel value
K(m, n)         = convolution-kernel weight
b               = bias
Y(i, j)         = output activation at location (i, j)
```

In simple terms:

```text
Take a small image region
        ↓
Multiply pixels by filter weights
        ↓
Add all multiplied values
        ↓
Add bias
        ↓
Produce one feature-map value
```

### Example Convolution

Consider the image region:

```text
1  2  3
4  5  6
7  8  9
```

and the filter:

```text
 1   0  -1
 1   0  -1
 1   0  -1
```

The convolution calculation is:

```text
(1×1) + (2×0) + (3×-1)
+
(4×1) + (5×0) + (6×-1)
+
(7×1) + (8×0) + (9×-1)
```

which becomes:

```text
1 - 3 + 4 - 6 + 7 - 9
= -6
```

That value becomes one location in the output feature map.

The filter then moves to another region of the image and repeats the calculation.

The convolutional filters begin as trainable numerical weights.

During training:

```text
Forward Pass
     ↓
Prediction
     ↓
Loss
     ↓
Backpropagation
     ↓
Gradients
     ↓
Optimizer Updates Filter Weights
```

As training continues, different filters become useful for detecting visual patterns such as:

```text
edges
textures
color transitions
curves
local shapes
```

Deeper convolutional layers combine these simpler patterns into more complex visual representations.

---

## ReLU Mathematics

The ReLU activation function is:

```text
ReLU(x) = max(0, x)
```

Equivalently:

```text
ReLU(x) = x     if x > 0
ReLU(x) = 0     if x ≤ 0
```

Examples:

```text
ReLU(-5) = 0
ReLU(-1) = 0
ReLU(0)  = 0
ReLU(3)  = 3
ReLU(8)  = 8
```

Conceptually:

```text
Negative activation
        ↓
Set to zero

Positive activation
        ↓
Keep the value
```

ReLU introduces non-linearity.

Without non-linear activation functions, stacking multiple linear layers would still result in a linear transformation.

---

## Max-Pooling Mathematics

Max pooling reduces the spatial size of a feature map by selecting the maximum value from a local region.

For example:

```text
1  4
2  7
```

a 2×2 max-pooling operation gives:

```text
max(1, 4, 2, 7) = 7
```

Therefore:

```text
1  4
2  7

 ↓ MaxPool

7
```

In this CNN:

```text
32 × 32
   ↓
16 × 16
```

and later:

```text
16 × 16
   ↓
8 × 8
```

Pooling reduces computation while retaining strong local feature responses.

---

## Linear Layer Mathematics

A linear neuron performs a weighted sum:

```text
z
=
w_1x_1
+
w_2x_2
+
...
+
w_nx_n
+
b
```

or more compactly:

```text
z = Σ_i (w_i × x_i) + b
```

where:

```text
x_i = input feature
w_i = trainable weight
b   = bias
z   = output
```

After convolution and pooling, the feature maps have shape:

```text
64 × 8 × 8
```

Flattening gives:

```text
64 × 8 × 8
=
4096 features
```

The classifier then transforms:

```text
4096 features
     ↓
Linear Layer
     ↓
128-dimensional representation
     ↓
ReLU
     ↓
Linear Layer
     ↓
10 class logits
```

The first linear layer can be expressed as:

```text
h = W_1 x + b_1
```

and the final classification layer as:

```text
z = W_2 h + b_2
```

where:

```text
x = flattened CNN features
h = 128-dimensional hidden representation
z = 10 output logits
```

---

## Cross-Entropy Loss Mathematics

The model is trained using **Cross-Entropy Loss**.

For a single example whose correct class is `y`, the loss is:

```text
L = -log(P_y)
```

where:

```text
P_y = predicted probability assigned to the correct class
```

Softmax first calculates:

```text
P_y
=
exp(z_y)
/
Σ_j exp(z_j)
```

Therefore the complete idea is:

```text
Logits
  ↓
Softmax
  ↓
Class Probabilities
  ↓
Probability of Correct Class
  ↓
-log(P_correct)
  ↓
Cross-Entropy Loss
```

For example:

```text
P_correct = 0.95
L = -log(0.95)
≈ 0.051
```

This is a small loss.

But:

```text
P_correct = 0.02
L = -log(0.02)
≈ 3.912
```

This is a much larger loss.

Therefore:

```text
High correct-class probability
        ↓
Small loss

Low correct-class probability
        ↓
Large loss
```

---

## Backpropagation Mathematics

After the forward pass, PyTorch calculates gradients using:

```python
loss.backward()
```

For every trainable parameter `w`, backpropagation computes:

```text
∂L / ∂w
```

This means:

```text
How much does the loss change
when this weight changes?
```

For a chain of transformations:

```text
x → a → b → L
```

the chain rule gives:

```text
∂L/∂x
=
(∂L/∂b)
×
(∂b/∂a)
×
(∂a/∂x)
```

In the CNN, gradients conceptually travel backward through:

```text
Loss
  ↓
Softmax / Cross-Entropy
  ↓
Final Linear Layer
  ↓
Hidden Linear Layer
  ↓
Second Convolution
  ↓
First Convolution
```

Each weight receives a gradient:

```text
gradient = ∂Loss / ∂Weight
```

This gradient tells the optimizer how the parameter should change to reduce the loss.

---

## Gradient Descent

The basic gradient-descent update is:

```text
w_new
=
w_old
-
η × (∂L / ∂w)
```

where:

```text
w = trainable parameter
L = loss
η = learning rate
```

For example:

```text
w_old       = 0.50
gradient    = 2.00
η           = 0.01
```

then:

```text
w_new
=
0.50 - (0.01 × 2.00)

=
0.48
```

The negative sign means the parameter moves in the direction that reduces the loss.

---

## Adam Optimization

This project uses the **Adam optimizer** rather than basic gradient descent.

Adam maintains moving estimates of:

```text
m_t = first moment of gradients
v_t = second moment of gradients
```

A simplified form is:

```text
m_t
=
β1 × m_(t-1)
+
(1 - β1) × g_t
```

and:

```text
v_t
=
β2 × v_(t-1)
+
(1 - β2) × g_t²
```

where:

```text
g_t = current gradient
β1  = first-moment decay factor
β2  = second-moment decay factor
```

Adam then uses these estimates to adapt the update size for different parameters.

The overall training process is:

```text
Images
  ↓
Forward Pass
  ↓
Logits
  ↓
Cross-Entropy Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Adam
  ↓
Parameter Update
```

---

## Epochs, Batches, and Training Steps

The CIFAR-10 training dataset contains:

```text
50,000 images
```

The batch size is:

```text
64 images
```

The approximate number of batches per epoch is:

```text
50,000 / 64
≈ 781.25
```

Because the final incomplete batch is also processed:

```text
≈ 782 batches per epoch
```

One batch typically results in one optimizer update.

Therefore:

```text
1 epoch
≈ 782 parameter updates
```

For 10 epochs:

```text
10 × 782
≈ 7,820 parameter updates
```

The training process can be visualized as:

```text
Batch 1
   ↓
Forward Pass
   ↓
Loss
   ↓
Backward Pass
   ↓
Update Weights

Batch 2
   ↓
Forward Pass
   ↓
Loss
   ↓
Backward Pass
   ↓
Update Weights

...

Batch 782
   ↓
Update Weights

================
Epoch 1 Complete
================

Then repeat for the next epoch.
```

---

## End-to-End Mathematical Pipeline

The complete CNN learning process can be summarized as:

```text
Input Image
X ∈ R^(3×32×32)

      ↓

Convolution

Y(i,j)
=
Σ_m Σ_n
[
X(i+m,j+n) × K(m,n)
]
+
b

      ↓

ReLU

ReLU(x) = max(0,x)

      ↓

Max Pooling

max(local region)

      ↓

More Convolutional Features

      ↓

Flatten

64 × 8 × 8
=
4096

      ↓

Linear Layer

h = W_1x + b_1

      ↓

128-D Representation

      ↓

Final Linear Layer

z = W_2h + b_2

      ↓

10 Logits

      ↓

Softmax

P_i
=
exp(z_i)
/
Σ_j exp(z_j)

      ↓

Cross-Entropy

L = -log(P_y)

      ↓

Backpropagation

∂L / ∂w

      ↓

Adam Optimization

w_new
=
w_old
-
adaptive_update

      ↓

Improved CNN Parameters
```

---

## Main Findings

### 1. CNNs Learn Hierarchical Visual Representations

The network progressively transforms raw image pixels into more useful representations.

```text
Pixels
 ↓
Edges and textures
 ↓
Feature combinations
 ↓
Higher-level representations
 ↓
Classification
```

---

### 2. Representation Quality Differs Across Classes

Some classes are represented more distinctly than others.

Strong-performing classes included:

```text
Automobile
Ship
Frog
Horse
```

More difficult classes included:

```text
Cat
Dog
Bird
Deer
```

---

### 3. Similar Visual Categories Are Frequently Confused

The strongest example was:

```text
Cat ↔ Dog
```

with:

```text
Cat → Dog = 167
Dog → Cat = 159
```

misclassifications.

---

### 4. Learned Representation Overlap Is Visible

The PCA analysis showed substantial overlap between multiple image classes, particularly among animal categories.

This observation is consistent with the confusion matrix.

---

### 5. Neural-Network Confidence Can Be Misleading

The network produced several incorrect predictions with high softmax confidence.

Therefore:

```text
Confidence ≠ Correctness
```

---

### 6. CNN Predictions Can Depend on Contextual Information

Grad-CAM showed that prediction-relevant activations can sometimes occur outside the most visually obvious object region.

The CNN may therefore use:

```text
Object information
        +
Background information
        +
Contextual visual patterns
```

when making a classification decision.

---

### 7. Accuracy Alone Does Not Explain Model Behavior

The final test accuracy was:

```text
71.62%
```

but accuracy alone does not explain:

```text
Which classes fail?
Why do they fail?
Which representations overlap?
How confident are incorrect predictions?
Which image regions influence predictions?
```

The confusion matrix, feature maps, PCA representation analysis, confidence analysis, and Grad-CAM provide a deeper view of the model.

---

## Project Structure

```text
visual-representation-analysis-cnn/
│
├── analysis/
│   ├── error_analysis.py
│   ├── feature_maps.py
│   ├── gradcam.py
│   ├── representation_analysis.py
│   └── training_summary.py
│
├── checkpoints/
│   └── cnn_cifar10.pth
│
├── data/
│
├── figures/
│   ├── confusion_matrix.png
│   ├── feature_original.png
│   ├── first_layer_feature_maps.png
│   ├── second_layer_feature_maps.png
│   ├── representation_pca.png
│   ├── wrong_predictions.png
│   ├── gradcam_example.png
│   ├── gradcam_correct_vs_wrong.png
│   ├── training_accuracy.png
│   └── training_loss.png
│
├── models/
│   └── cnn.py
│
├── training/
│   ├── train.py
│   └── evaluate.py
│
├── utils/
│
├── 01_tensor_basics.py
├── 02_autograd_basics.py
├── 03_dataloader_basics.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Requirements

The project requires:

```text
torch>=2.0
torchvision>=0.15
matplotlib>=3.7
scikit-learn>=1.3
numpy>=1.24
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aacaas5/visual-representation-analysis-cnn.git
```

Enter the project directory:

```bash
cd visual-representation-analysis-cnn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### PyTorch Tensor Basics

```bash
python 01_tensor_basics.py
```

### Autograd Basics

```bash
python 02_autograd_basics.py
```

### DataLoader Basics

```bash
python 03_dataloader_basics.py
```

### Train the CNN

```bash
python training/train.py
```

### Evaluate the CNN

```bash
python training/evaluate.py
```

### Generate Confusion Matrix and Error Analysis

```bash
python analysis/error_analysis.py
```

### Visualize Feature Maps

```bash
python analysis/feature_maps.py
```

### Analyze Learned Representations Using PCA

```bash
python analysis/representation_analysis.py
```

### Generate Grad-CAM Explanations

```bash
python analysis/gradcam.py
```

### Generate Training Curves

```bash
python analysis/training_summary.py
```

---

## Complete Learning Pipeline

```text
PyTorch Tensors
      ↓
Autograd
      ↓
DataLoader
      ↓
CIFAR-10 Images
      ↓
CNN Architecture
      ↓
Convolution
      ↓
ReLU
      ↓
Max Pooling
      ↓
Feature Extraction
      ↓
Flattening
      ↓
128-D Representation
      ↓
Classification Logits
      ↓
Softmax
      ↓
Cross-Entropy Loss
      ↓
Backpropagation
      ↓
Adam Optimization
      ↓
Image Classification
      ↓
Test Evaluation
      ↓
Confusion Matrix
      ↓
Feature-Map Visualization
      ↓
Representation Analysis
      ↓
PCA
      ↓
Confidence Analysis
      ↓
Grad-CAM
      ↓
Correct-vs-Incorrect Analysis
```

---

## Conclusion

This project demonstrates that evaluating a convolutional neural network only through classification accuracy provides an incomplete picture of model behavior.

The compact CNN achieved:

```text
Training accuracy: 84.45%
Test accuracy:     71.62%
```

on CIFAR-10.

However, deeper analysis revealed:

- significant differences in class-level performance,
- systematic confusion between visually similar categories,
- overlapping learned representations,
- highly confident incorrect predictions,
- and reliance on localized or contextual visual features.

The experiments show that understanding a CNN requires more than measuring its final accuracy.

By combining:

```text
Classification Performance
        +
Confusion Analysis
        +
Feature-Map Visualization
        +
Representation-Space Analysis
        +
Confidence Analysis
        +
Grad-CAM
```

the project provides a broader view of how convolutional neural networks transform visual information and use learned representations to make classification decisions.

---

## Repository Description

CNN representation analysis on CIFAR-10 using PyTorch, feature maps, PCA, confidence analysis, confusion matrices, and Grad-CAM.
