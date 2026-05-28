import pandas as pd
import torch
import torch.nn as nn
df = pd.read_csv('../data/cnn_data.csv')
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
    def forward(self, x):
        return self.conv1(x)
model = SimpleCNN()
print("CNN Created!")
