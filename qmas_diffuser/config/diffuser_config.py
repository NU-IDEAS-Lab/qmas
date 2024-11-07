import diffuser.models.base_temporal_models as base_models
import diffuser.diffuser_models as diffuser_models
from diffuser.datasets.normalization import GaussianNormalizer
from diffuser.datasets.sequence import SequenceDataset

from diffuser.utils.setup import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]
# model
base_model = base_models.TemporalUnet
diffusion = diffuser_models.GaussianDiffusion
horizon = 32
n_diffusion_steps = 20
action_weight = 10
loss_weights =  None
loss_discount = 1
predict_epsilon = False
dim_mults = (1, 2, 4, 8)
attention = False

## dataset
loader = SequenceDataset
normalizer = GaussianNormalizer
preprocess_fns = []
clip_denoised = False
use_padding = True
max_path_length = 1000

## serialization
logbase = logbase
prefix = diffusion/defaults
exp_name = watch(args_to_watch)

## training
n_steps_per_epoch = 10000
loss_type = 'l2'
n_train_steps = 1e6
batch_size = 32
learning_rate = 2e-4
gradient_accumulate_every = 2
ema_decay = 0.995
save_freq = 20000
sample_freq = 20000
n_saves = 5
save_parallel = False
n_reference = 8
bucket = None
device = 'cuda'
seed =  None