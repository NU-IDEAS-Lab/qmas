import argparse
import diffuser.utils.training.Trainer as Trainer
import diffuser.config.diffuser_config as diffuser_config
import pdb

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

parser = argparse.ArgumentParser(description='Training script with customizable environment')
parser.add_argument('--env_name', type=str, default='hopper-medium-expert-v2',
                  help='Environment name for training')
args = parser.parse_args()


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = diffuser_config.loader(
    # savepath='dataset_config.pkl',
    env=diffuser_config.dataset,
    horizon=diffuser_config.horizon,
    normalizer=diffuser_config.normalizer,
    preprocess_fns=diffuser_config.preprocess_fns,
    use_padding=diffuser_config.use_padding,
    max_path_length=diffuser_config.max_path_length,
)

dataset = dataset_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = diffuser_config.base_model(
    # savepath='model_config.pkl',
    horizon=diffuser_config.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=diffuser_config.dim_mults,
    attention=diffuser_config.attention,
    device=diffuser_config.device,
)

diffusion_config = diffuser_config.diffusion_model(
    # savepath='diffusion_config.pkl',
    horizon=diffuser_config.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=diffuser_config.n_diffusion_steps,
    loss_type=diffuser_config.loss_type,
    clip_denoised=diffuser_config.clip_denoised,
    predict_epsilon=diffuser_config.predict_epsilon,
    ## loss weighting
    action_weight=diffuser_config.action_weight,
    loss_weights=diffuser_config.loss_weights,
    loss_discount=diffuser_config.loss_discount,
    device=diffuser_config.device,
)

trainer_config = Trainer(
    # savepath='trainer_config.pkl',
    train_batch_size=diffuser_config.batch_size,
    train_lr=diffuser_config.learning_rate,
    gradient_accumulate_every=diffuser_config.gradient_accumulate_every,
    ema_decay=diffuser_config.ema_decay,
    sample_freq=diffuser_config.sample_freq,
    save_freq=diffuser_config.save_freq,
    label_freq=int(diffuser_config.n_train_steps // diffuser_config.n_saves),
    save_parallel=diffuser_config.save_parallel,
    bucket=diffuser_config.bucket,
    n_reference=diffuser_config.n_reference,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(diffuser_config.n_train_steps // diffuser_config.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs}')
    trainer.train(n_train_steps=diffuser_config.n_steps_per_epoch)

