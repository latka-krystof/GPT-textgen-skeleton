import torch
import torch.nn as nn
import torch.nn.functional as F
import constants

# import constants
n_embd = constants.N_EMBD
block_size = constants.BLOCK_SIZE
dropout = constants.DROPOUT
n_head = constants.N_HEAD
n_layer = constants.N_LAYER

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        return x


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()

    def forward(self, x):
        return x


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()

    def forward(self, x):
        return x


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

    def forward(self, x):
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

    def _init_weights(self, module):
        pass

    def forward(self, idx, targets=None, device='cpu'):
        logits, loss = None, None
        return logits, loss

    def generate(self, idx, max_new_tokens, device='cpu'):
        return idx