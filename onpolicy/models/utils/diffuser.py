from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

from diffuser_helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses
)

Sample = namedtuple('Sample', 'trajectories values chains')

# class LayerNorm(nn.Module):
#     def __init__(self, dim, eps = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.g = nn.Parameter(torch.ones(1, dim, 1))
#         self.b = nn.Parameter(torch.zeros(1, dim, 1))

#     def forward(self, x):
#         var = torch.var(x, dim=1, unbiased=False, keepdim=True)
#         mean = torch.mean(x, dim=1, keepdim=True)
#         return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = LayerNorm(dim)

#     def forward(self, x):
#         x = self.norm(x)
#         return self.fn(x)
    
# class Conv1dBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super().__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
#         self.activation = nn.Mish()
    
#     def forward(self, x):
#         return self.activation(self.conv(x))

# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         device = x.device
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb
    
class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon = 32, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class TemporalUnet(nn.Module):
    def __init__(
        self,
        horizon = 32,          # Length of input sequence
        transition_dim,   # Dimension of input/output features
        dim=32,          # Base dimension for the network
        dim_mults=(1, 2, 4, 8),  # Dimension multipliers for each level
    ):
        super().__init__()
        
        # Calculate dimensions for each layer
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        
        # Downsampling path
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            if not is_last:
                horizon = horizon // 2
        
        # Middle blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim, horizon=horizon)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim, horizon=horizon)
        
        # Upsampling path
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=dim, horizon=horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
            if not is_last:
                horizon = horizon * 2
        
        # Final convolution
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time):
        # Reshape input: [batch, horizon, transition] -> [batch, transition, horizon]
        x = einops.rearrange(x, 'b h t -> b t h')
        
        # Time embedding
        t = self.time_mlp(time)
        
        # Store skip connections
        h = []
        
        # Downsampling
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)
        
        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        # Upsampling with skip connections
        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)
        
        # Final convolution and reshape back
        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon = 32, observation_dim, action_dim, n_timesteps=20,
        loss_type='l2', clip_denoised=False, predict_epsilon=False,
        action_weight=10, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = self.model = TemporalUnet(
            horizon=horizon,
            transition_dim=transition_dim, 
            dim=32, 
            dim_mults=(1, 2, 4, 8)  
        )

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight = 10, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)