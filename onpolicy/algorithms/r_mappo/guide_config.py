# Model settings
model = models.ValueFunction()
diffusion = models.ValueDiffusion()
horizon = 32
n_diffusion_steps = 20
dim_mults = (1, 2, 4, 8)
# renderer = 'utils.MuJoCoRenderer'

# Value-specific settings
discount = 0.99
termination_penalty = -100
normed = False

# Dataset settings
# loader = 'datasets.ValueDataset'
normalizer = 'GaussianNormalizer'
preprocess_fns = []
use_padding = True
max_path_length = 1000

# Serialization settings
logbase = None # Need to set this
prefix = 'values/defaults'
exp_name = None # Need to implement watch(args_to_watch)

# Training settings
n_steps_per_epoch = 10000
loss_type = 'value_l2'
n_train_steps = int(200e3)
batch_size = 32
learning_rate = 2e-4
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