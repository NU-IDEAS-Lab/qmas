import argparse
import diffuser.utils.training.Trainer as Trainer
import diffuser.config.guide_config as guide_config
import pdb
import pickle

parser = argparse.ArgumentParser(description='Training script with customizable environment')
parser.add_argument('--env_name', type=str, default='walker2d-medium-replay-v2',
                  help='Environment name for training')
args = parser.parse_args()

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#
def pickle_dump_savepath(savepath):
    if savepath:
        pickle.dump(open(savepath, 'wb'))
        print(f'[ utils/config ] Saved config to: {savepath}\n')
    return savepath

dataset_config = guide_config.loader(
    # savepath=pickle_dump_savepath('dataset_config.pkl'),
    env=env_name,
    horizon=guide_config.horizon,
    normalizer=guide_config.normalizer,
    preprocess_fns=guide_config.preprocess_fns,
    use_padding=guide_config.use_padding,
    max_path_length=guide_config.max_path_length,
    ## value-specific kwargs
    discount=guide_config.discount,
    termination_penalty=guide_config.termination_penalty,
    normed=guide_config.normed,
)


dataset = dataset_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = guide_config.base_model(
    # savepath='model_config.pkl',
    horizon=guide_config.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=guide_config.dim_mults,
    device=guide_config.device,
)

diffusion_config = guide_config.diffusion_model(
    # savepath='diffusion_config.pkl',
    horizon=guide_config.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=guide_config.n_diffusion_steps,
    loss_type=guide_config.loss_type,
    device=guide_config.device,
)

trainer_config = Trainer(
    # savepath='trainer_config.pkl',
    train_batch_size=guide_config.batch_size,
    train_lr=guide_config.learning_rate,
    gradient_accumulate_every=guide_config.gradient_accumulate_every,
    ema_decay=guide_config.ema_decay,
    sample_freq=guide_config.sample_freq,
    save_freq=guide_config.save_freq,
    label_freq=int(guide_config.n_train_steps // guide_config.n_saves),
    save_parallel=guide_config.save_parallel,
    bucket=guide_config.bucket,
    n_reference=guide_config.n_reference,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])

loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(guide_config.n_train_steps // guide_config.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs}')
    trainer.train(n_train_steps=guide_config.n_steps_per_epoch)
