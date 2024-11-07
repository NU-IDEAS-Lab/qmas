import diffuser.models.base_temporal_models as base_models
import diffuser.diffuser_models as diffuser_models
from diffuser.utils.setup import watch
from diffuser.datasets.normalization import GaussianNormalizer
from diffuser.datasets.sequence import SequenceDataset

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

model = base_models.ValueFunction,
diffusion = diffuser_models.ValueDiffusion
horizon = 32
n_diffusion_steps = 20
dim_mults = (1, 2, 4, 8)

## value-specific kwargs
discount = 0.99
termination_penalty = -100
normed = False

## dataset
loader = datasets.ValueDataset
normalizer = GaussianNormalizer
preprocess_fns = []
use_padding = True
max_path_length = 1000

## serialization
logbase = logbase
prefix = values/defaults
exp_name = watch(args_to_watch)

## training
n_steps_per_epoch = 10000
loss_type = 'value_l2'
n_train_steps: 200e3
batch_size =  = 32
learning_rate: 2e-4
gradient_accumulate_every = 2
ema_decay = 0.995
save_freq = 1000
sample_freq = 0
n_saves = 5
save_parallel = False
n_reference = 8
bucket = None
device = 'cuda'
seed = None