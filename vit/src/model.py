import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math

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


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.model.num_heads
        self.hidden_dims = config.model.hidden_size
        self.head_size = self.hidden_dims // self.num_heads
        self.all_head_size = int(self.num_heads * self.head_size)
        assert self.hidden_dims % self.num_heads == 0


        self.query = nn.Linear(self.hidden_dims, self.all_head_size)
        self.key = nn.Linear(self.hidden_dims, self.all_head_size)
        self.value = nn.Linear(self.hidden_dims, self.all_head_size)
        self.out = nn.Linear(self.hidden_dims, self.all_head_size)

        self.attn_dropout = nn.Dropout(p = config.model.attn_dropout_rate)
        

    def forward(self, x):
        B, N, D = x.shape
        
        Q = self.query(x).view(B, N, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        K = self.key(x).view(B, N, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        V = self.value(x).view(B, N, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_size)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.hidden_dims)

        out = self.out(attn_out)
        return out, attn_weights
