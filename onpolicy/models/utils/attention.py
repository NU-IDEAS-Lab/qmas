import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

class SelfAttention(nn.Module):
    ''' Taken in large part from https://medium.com/@heyamit10/implement-self-attention-and-cross-attention-in-pytorch-cfe17ab0b3ee '''
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        # Define linear transformations for Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        # Generate Q, K, V matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Calculate attention using our scaled dot-product function
        out = scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        return out
