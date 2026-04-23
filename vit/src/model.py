import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.model.patch_size
        self.hidden_dim = config.model.hidden_size
        self.in_channels = config.model.channels
        self.image_size = config.model.image_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        

        self.patch_embeddings = nn.Conv2d(
                in_channels = self.in_channels,
                out_channels = self.hidden_dim,
                kernel_size = self.patch_size,
                stride = self.patch_size
            )

        self.positional_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.dropout = nn.Dropout(p=config.model.dropout_rate)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.positional_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
