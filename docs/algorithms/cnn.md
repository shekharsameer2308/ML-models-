# Convolutional Neural Networks (CNN)

## Overview
A Convolutional Neural Network (CNN) is a class of deep neural networks, most commonly applied to analyzing visual imagery. They are inspired by the biological processes of the visual cortex.

## Type
- Deep Learning
- Classification, Regression, Object Detection, Segmentation

## How it Works
CNNs use spatial correlations in data. They employ mathematical operations called "convolutions" in place of general matrix multiplication in at least one of their layers. Filters (or kernels) slide across the input image to extract features like edges, corners, and later, more complex patterns. A typical architecture consists of Convolutional layers, Pooling layers (to reduce spatial dimensions), and Fully Connected layers at the end for the final prediction.

## Pros and Cons
**Pros:**
- State-of-the-art performance on image and video recognition.
- Parameter sharing and sparse connections make them much more efficient than MLPs for images.
- Automatically learns spatial hierarchies of features.

**Cons:**
- Requires a massive amount of labeled data to train from scratch.
- Computationally very expensive, often requiring GPUs for training.
- Less effective on non-grid structured data (like tabular data).

## Common Use Cases
- Image classification.
- Object detection and localization.
- Facial recognition.
- Medical image analysis.

## Code Example (PyTorch Snippet)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 6, 3)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = SimpleCNN()
print(net)
```
