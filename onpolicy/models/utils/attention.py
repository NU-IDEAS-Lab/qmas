import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from onpolicy.models.utils.util import init as util_init

class SelfAttention(nn.Module):
    ''' Taken in large part from https://medium.com/@heyamit10/implement-self-attention-and-cross-attention-in-pytorch-cfe17ab0b3ee '''
    def __init__(self, embed_size, use_orthogonal: bool = True, use_ReLU: bool = True):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        # Define linear transformations for Q, K, V
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])
        def _init_(m):
            return util_init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        self.query = _init_(nn.Linear(embed_size, embed_size))
        self.key = _init_(nn.Linear(embed_size, embed_size))
        self.value = _init_(nn.Linear(embed_size, embed_size))

    def forward(self, x, mask=None):
        # Generate Q, K, V matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Calculate attention using our scaled dot-product function
        out = scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        return out
