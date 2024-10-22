import diffuser.utils as utils
import diffuser.models as models

# Model settings
model = models.TemporalUnet
diffusion = models.GaussianDiffusion
horizon = 32
n_diffusion_steps = 20
action_weight = 10
loss_weights = None  
loss_discount = 1
predict_epsilon = False
dim_mults = (1, 2, 4, 8)
attention = False
# renderer = 'utils.MuJoCoRenderer'

# Dataset settings
# loader = 'datasets.SequenceDataset'
normalizer = 'GaussianNormalizer' 
preprocess_fns = []
clip_denoised = False
use_padding = True
max_path_length = 1000

# Serialization settings
logbase = None # Need to set this
prefix = 'diffusion/defaults'
exp_name = None # Need to implement watch(args_to_watch)

# Training settings
n_steps_per_epoch = 10000
loss_type = 'l2'
n_train_steps = int(1e6)
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
seed = None