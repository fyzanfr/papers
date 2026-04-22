import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.model.patch_size
        self.hidden_dim = config.model.hidden_size
        self.in_channels = config.model.channels

        self.projection = nn.Conv2d(
                in_channels = self.in_channels,
                out_channels = self.hidden_dim,
                kernel_size = self.patch_size,
                stride = self.patch_size
            )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

    return x
