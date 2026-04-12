from collections import defaultdict, deque
from collections.abc import Iterable
import time
import einops

import imageio
import numpy as np
import torch
import wandb
import zarr
from tqdm.auto import tqdm
from collections import deque

from onpolicy.utils.util import get_shape_from_act_space, get_shape_from_obs_space
from onpolicy.runner.shared.base_runner import Runner

from onpolicy.utils.shared_buffer_torchrl import SharedReplayBuffer
from onpolicy.utils.trajectory_buffer import TrajectoryBuffer

from isru_zoo.env.isru_env import parallel_env_map_obs as pemo


class PettingzooRunner(Runner):
    def __init__(self, config):

        # The default restore functionality is broken. Disable it and do it ourselves.
        model_dir = config['all_args'].model_dir
        config['all_args'].model_dir = None

        super(PettingzooRunner, self).__init__(config)

        # Override the default replay buffer with our new TorchRL one.
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

        # Set up a trajectory buffer for use during training.
        self.train_trajectory = TrajectoryBuffer(
            history_length=self.all_args.prediction_history_window,
            transition_includes_actions=self.all_args.prediction_history_include_actions
        )
        self.train_prediction_prev = None

        self.env_infos = defaultdict(list)
       
        # Perform restoration.
        config['all_args'].model_dir = model_dir
        self.model_dir = config['all_args'].model_dir
        if self.model_dir is not None:
            self.restore(self.model_dir)
        
        if self.all_args.torch_compile:
            self.train_compiled = torch.compile(self.train, fullgraph=False)

    def run(self):
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_save_step = 0

        for episode in (progress_bar := tqdm(range(episodes), dynamic_ncols=True)):
            start_episode = time.time()

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            # Reset the environment and perform warmup.
            self.trainer.prep_rollout()
            self.warmup()

            # Set the delta steps to 1.
            delta_steps = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
            for step in range(self.episode_length):
                # Sample actions, collect values and probabilities.
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, uncertainty = self.collect(step)
                
                # Take a step in the environment and get the results.
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                # Get the number of steps taken by each agent since the agent was last ready.
                delta_steps = np.array([info["deltaSteps"] for info in infos])

                # insert data into buffer
                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, delta_steps, available_actions, uncertainty
                self.insert(data)

            # Get certain stats.
            avg_episode_rewards = self.buffer.rewards.mean().item() * self.episode_length

            # compute return
            if self.policy.IS_TRAINABLE:
                self.compute()

            if self.all_args.torch_compile:
                train_infos = self.train_compiled()
            else:
                train_infos = self.train(episode=episode, episodes=episodes)

            end = time.time()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model at every interval
            if episode == episodes - 1 or total_num_steps - last_save_step >= self.save_interval:
                self.save(episode)
                last_save_step = total_num_steps

            # log information
            if total_num_steps % self.log_interval == 0:
                
                train_infos["average_episode_rewards"] = avg_episode_rewards
                train_infos["fps"] = self.episode_length * self.n_rollout_threads / (end - start_episode)
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = defaultdict(list)

            progress_bar.set_postfix({
                "exp": self.experiment_name,
                "timesteps": f"{total_num_steps}/{self.num_env_steps}",
                "avg_ep_rewards": avg_episode_rewards,
                "fps": int(total_num_steps / (end - start))
            })

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # Reset environment.
        obs, share_obs, available_actions, infos = self.envs.reset(return_info=True)

        # Get the shape of the action space.
        act_shape = get_shape_from_act_space(self.buffer.act_space)
        if isinstance(act_shape, Iterable):
            actions_shape = (self.n_rollout_threads, self.num_agents, *act_shape)
        else:
            actions_shape = (self.n_rollout_threads, self.num_agents, act_shape)
        
        # Get the shape of action log probabilities from the policy.
        if self.policy.IS_TRAINABLE:
            action_log_prob_shape = (self.n_rollout_threads, self.num_agents, self.policy.actor.act.log_prob_dim)
        else:
            action_log_prob_shape = (self.n_rollout_threads, self.num_agents, 1)

        observation_global = None
        if "observation_global" in infos[0]:
            observation_global = np.array([info["observation_global"] for info in infos])

        # Initialize buffer.
        self.buffer.insert(
            share_obs=share_obs,
            obs=obs,
            rnn_states_actor=np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32),
            rnn_states_critic=np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32),
            actions=np.zeros(actions_shape, dtype=np.float32),
            action_log_probs=np.zeros(action_log_prob_shape, dtype=np.float32),

            # Although there is actually only one value per thread, we store it as if there were one per agent.
            # Each agent will have the same value.
            value_preds=np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32),

            rewards=np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32),
            masks=np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32),
            delta_steps=np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32),
            available_actions=available_actions,
            observation_global=observation_global,
        )

        # Reset and seed training trajectory buffer with the initial observations.
        act_size = np.prod(get_shape_from_act_space(self.buffer.act_space))
        self.train_prediction_prev = None
        self.train_trajectory.reset()
        # self.train_trajectory.add(
        #     obs=torch.from_numpy(obs).to(self.device),
        #     visibility_mask=torch.ones(obs.shape, dtype=torch.float32, device=self.device),
        #     action=torch.zeros(act_size, dtype=torch.float32, device=self.device)
        # )


    @torch.no_grad()
    def collect(self, step):
        share_obs, obs, global_obs, rnn_states, rnn_states_critic, masks, available_actions = self.buffer.compatibility_get_policy_input(step)

        uncertainty_now = None

        # If a predictor is available, replace obs with predictions from the trajectory buffers.
        use_prediction = (
            hasattr(self.policy, "predictors")
            and not self.all_args.prediction_disable
            and self.all_args.episode_fraction_start_prediction <= (step * self.episode_length * self.n_rollout_threads) / self.num_env_steps
            and self.train_trajectory.ready()
            and (self.all_args.prediction_during_training or self.all_args.prediction_uq_injection_method != "none")
        )
        if use_prediction:
            # Get the prediction and the uncertainty.
            trajectory, visibility_mask = self.train_trajectory.get_trajectory()
            trajectory = trajectory.transpose(0, 1)
            visibility_mask = visibility_mask.transpose(0, 1)
            pred, uncertainty = self.trainer.policy.get_prediction(
                trajectory=trajectory,
                visibility_mask=visibility_mask,
                prediction_prev=self.train_prediction_prev,
                has_sample_dim=True
            )
            pred = pred.detach().cpu()
            self.train_prediction_prev = pred
            uncertainty = uncertainty.detach().cpu()

            # Get the currrent state.
            pred_now = pred[:, -1]
            uncertainty_now = uncertainty[:, -1]

            # Replace the observation with the prediction.
            if self.all_args.prediction_during_training:
                obs = pred_now
            
            # Inject uncertainty into the observation.
            if self.all_args.prediction_uq_injection_method == "append":
                if global_obs is not None and self.all_args.state_encoder:
                    global_obs = torch.cat([global_obs, uncertainty_now], dim=-1)
                elif obs.dtype == object:
                    raise ValueError("Must have observation_global in infos to use object-type observations (e.g. pyg graph observations). Check that environment is providing this information.")
                else:
                    obs = torch.cat([obs, uncertainty_now], dim=-1)
        
        # Handle case where UQ injection expected, but not yet available due to predictor not running.
        elif self.all_args.prediction_uq_injection_method == "append":
            uncertainty_now = None
            if global_obs is not None and self.all_args.state_encoder:
                uncertainty_now = torch.zeros_like(global_obs)
                global_obs = torch.cat([global_obs, uncertainty_now], dim=-1)
            elif obs.dtype == object:
                raise ValueError("Must have observation_global in infos to use object-type observations (e.g. pyg graph observations). Check that environment is providing this information.")
            else:
                uncertainty_now = torch.zeros_like(obs)
                obs = torch.cat([obs, uncertainty_now], dim=-1)

        values, action, action_log_prob, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            masks,
            available_actions=available_actions,
            global_obs=global_obs
        )

        # The value function predictions are made once over the entire share_obs.
        # However, we need to compare them with per-agent returns.
        # This is done by repeating the value predictions for each agent.
        values = values.detach().cpu().reshape((self.n_rollout_threads, 1, 1)).repeat(1, self.policy.args.num_agents, 1)

        actions = action.detach().cpu().reshape((self.n_rollout_threads, self.num_agents, *action.shape[1:]))
        action_log_probs = action_log_prob.detach().cpu().reshape((self.n_rollout_threads, self.num_agents, *action_log_prob.shape[1:]))
        rnn_states = rnn_states.detach().cpu().reshape((self.n_rollout_threads, self.num_agents, *rnn_states.shape[1:]))
        rnn_states_critic = rnn_states_critic.detach().cpu().reshape((self.n_rollout_threads, self.num_agents, *rnn_states_critic.shape[1:]))

        # if actions.shape[-1] == 1:
        #     actions_env = [actions[idx, :, 0].numpy() for idx in range(self.n_rollout_threads)]
        # else:
        #     actions_env = [actions[idx, :, :].numpy() for idx in range(self.n_rollout_threads)]
        actions_env = [actions[idx, :, :].numpy() for idx in range(self.n_rollout_threads)]

        if uncertainty_now is not None:     
            uncertainty_now = uncertainty_now.cpu().reshape((self.n_rollout_threads, self.num_agents, *uncertainty_now.shape[1:]))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, uncertainty_now

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, delta_steps, available_actions, uncertainty = data
        
        # update env_infos if done
        dones_env = np.all(dones, axis=-1)

        # Get visibility mask from infos.
        visibility_mask = None
        if "visibility_mask" in infos[0]:
            visibility_mask = np.array([info["visibility_mask"] for info in infos])
        
        observation_global = None
        if "observation_global" in infos[0]:
            observation_global = np.array([info["observation_global"] for info in infos])

        # Get extra state information from infos.
        state_visibility_mask = None
        if "state_visibility_mask" in infos[0]:
            state_visibility_mask = np.array([info["state_visibility_mask"] for info in infos])

        # Add information to the logger.
        keys = infos[0].keys()
        for key in keys:
            if type(key) == str:
                self.env_infos[key] = [i[key] for i in infos]

        masks = torch.ones((self.n_rollout_threads, self.num_agents, 1))
        for i in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                if dones[i, agent_id]:
                    rnn_states[i][agent_id] = torch.zeros((self.recurrent_N, self.hidden_size))
                    rnn_states_critic[i][agent_id] = torch.zeros((self.recurrent_N, self.hidden_size))
                    masks[i, agent_id] = torch.zeros(1)

        self.buffer.insert(
            share_obs=share_obs,
            obs=obs,
            rnn_states_actor=rnn_states,
            rnn_states_critic=rnn_states_critic,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            rewards=rewards,
            masks=masks,
            delta_steps=delta_steps,
            available_actions=available_actions,
            visibility_mask=visibility_mask,
            state_visibility_mask=state_visibility_mask,
            observation_global=observation_global,
            observation_uncertainty=uncertainty,
        )

        # Update the trajectory buffer with the new transition.
        self._update_trajectory_buffer(
            self.train_trajectory,
            obs,
            infos,
            actions,
            self.n_rollout_threads,
            share_obs=share_obs
        )


    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""

        share_obs, obs, global_obs, rnn_states, rnn_states_critic, masks, available_actions = self.buffer.compatibility_get_policy_input(-1)

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer.policy.get_values(share_obs,
                                                        obs,
                                                        rnn_states_critic,
                                                        masks)
        else:
            next_values = self.trainer.policy.get_values(share_obs,
                                                        rnn_states_critic,
                                                        masks)
        next_values = next_values.detach().cpu().view(self.n_rollout_threads, 1, 1)
        next_values = next_values.repeat(1, self.num_agents, 1)
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)


    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            # if type(v) == wandb.viz.CustomChart and self.use_wandb:
            #     wandb.log({k: v}, step=total_num_steps)
            # elif len(v) > 0:
            if len(v) > 0:
                if isinstance(v[0], np.ndarray):
                    v = np.array(v)
                
                    # Don't log large matrices.
                    if v.ndim > 2:
                        continue

                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)    

    def _build_viz_mask(self, obs, infos, actions, n_threads):
        """Build the combined visibility mask for observations (and optionally actions).
        
        Applies the observation mask to obs in-place when configured. Returns viz_mask.
        """
        viz_mask_obs = np.array([info["visibility_mask"] for info in infos]) if "visibility_mask" in infos[0] else np.ones_like(obs)
        if self.all_args.observation_mask:
            obs = obs * viz_mask_obs

        if self.all_args.prediction_history_include_actions:
            viz_mask_actions = np.ones(actions.shape, dtype=np.float32)
            viz_mask = np.concatenate(
                [viz_mask_actions, viz_mask_obs.reshape(n_threads, self.num_agents, -1)], axis=-1
            )
        else:
            viz_mask = viz_mask_obs.reshape(n_threads, self.num_agents, -1)
        return viz_mask, obs

    def _seed_trajectory_buffer(self, trajectory_buffer, obs, n_threads, share_obs=None):
        # _obs = share_obs if (obs.dtype == object and share_obs is not None) else obs
        # traj_obs = einops.rearrange(_obs, "t n ... -> (t n) ...")
        # traj_visibility_mask = np.ones_like(traj_obs, dtype=np.float32)
        # act_shape = get_shape_from_act_space(self.buffer.act_space)
        # if isinstance(act_shape, Iterable):
        #     traj_actions_shape = (n_threads * self.num_agents, *act_shape)
        # else:
        #     traj_actions_shape = (n_threads * self.num_agents, act_shape)
        # traj_actions = torch.zeros(traj_actions_shape, dtype=torch.float32, device=self.device)

        trajectory_buffer.reset()
        # trajectory_buffer.add(
        #     obs=torch.from_numpy(traj_obs).to(self.device),
        #     visibility_mask=torch.from_numpy(traj_visibility_mask).to(self.device),
        #     action=traj_actions,
        # )

    def _update_trajectory_buffer(self, trajectory_buffer, obs, infos, actions, n_threads, share_obs=None):
        # Get data from infos.
        observation_global = np.array([info["observation_global"] for info in infos]) if "observation_global" in infos[0] else None
        state_visibility_mask = np.array([info["state_visibility_mask"] for info in infos]) if "state_visibility_mask" in infos[0] else None
        visibility_mask = np.array([info["visibility_mask"] for info in infos]) if "visibility_mask" in infos[0] else None
        visibility_mask_global = np.array([info["visibility_mask_global"] for info in infos]) if "visibility_mask_global" in infos[0] else None

        # Select appropriate vectors to add to the trajectory buffer.
        if observation_global is not None and self.all_args.state_encoder:
            traj_obs = observation_global
            traj_visibility_mask = visibility_mask_global
        elif obs.dtype == object:
            raise ValueError("Must have observation_global in infos to use object-type observations (e.g. pyg graph observations). Check that environment is providing this information.")
        else:
            traj_obs = obs
            traj_visibility_mask = visibility_mask
        assert traj_visibility_mask is not None, "Must have some form of visibility mask in infos to update trajectory buffer. Check that environment is providing this information."

        traj_obs = einops.rearrange(traj_obs, "t n ... -> (t n) ...")
        traj_visibility_mask = einops.rearrange(traj_visibility_mask, "t n ... -> (t n) ...")
        traj_actions = einops.rearrange(actions.detach().cpu(), "t n ... -> (t n) ...")

        trajectory_buffer.add(
            obs=torch.from_numpy(traj_obs).to(self.device),
            visibility_mask=torch.from_numpy(traj_visibility_mask).to(self.device),
            action=traj_actions.to(self.device),
        )

    def _get_prediction_target(self, obs, infos, n_threads, share_obs=None):
        """Get the full target tensor and visibility mask for the current prediction timestep."""

        observation_global = np.array([info["observation_global"] for info in infos]) if "observation_global" in infos[0] else None
        visibility_mask = np.array([info["visibility_mask"] for info in infos]) if "visibility_mask" in infos[0] else None
        visibility_mask_global = np.array([info["visibility_mask_global"] for info in infos]) if "visibility_mask_global" in infos[0] else None
        state_visibility_mask = np.array([info["state_visibility_mask"] for info in infos]) if "state_visibility_mask" in infos[0] else None

        if observation_global is not None and self.all_args.state_encoder:
            target = observation_global
            target_visibility_mask = visibility_mask_global
        elif share_obs is not None:
            target = np.array(share_obs)
            target_visibility_mask = state_visibility_mask
        elif obs.dtype == object:
            raise ValueError("Must have observation_global in infos to use object-type observations (e.g. pyg graph observations). Check that environment is providing this information.")
        else:
            target = obs
            target_visibility_mask = visibility_mask

        if target_visibility_mask is None:
            target = np.array(target)
            target_visibility_mask = np.ones_like(target, dtype=np.float32)

        target = target.reshape((n_threads, self.num_agents, -1))
        target_visibility_mask = target_visibility_mask.reshape((n_threads, self.num_agents, -1))
        return target, target_visibility_mask

    def _compute_prediction_errors(self, prediction, target, target_visibility_mask, n_threads, obs_shape):
        """Compute prediction error on hidden entries and over the full target tensor."""

        pred_now = prediction[:, -1].detach().cpu().reshape((n_threads, self.num_agents, *obs_shape))
        target_now = torch.from_numpy(target).to(dtype=pred_now.dtype).reshape((n_threads, self.num_agents, *obs_shape))
        visibility_now = torch.from_numpy(target_visibility_mask).to(dtype=pred_now.dtype).reshape((n_threads, self.num_agents, *obs_shape))

        full_error = torch.abs(pred_now - target_now)
        hidden_mask = 1.0 - visibility_now
        hidden_count = hidden_mask.sum()
        if hidden_count.item() > 0:
            hidden_error_mean = (full_error * hidden_mask).sum() / hidden_count
        else:
            hidden_error_mean = torch.zeros((), dtype=pred_now.dtype)

        return hidden_error_mean, full_error

    def _compute_predictions(self, trajectory_buffer, prediction_prev, n_threads, obs_shape):
        use_prediction = (
            hasattr(self.policy, "predictors")
            and trajectory_buffer.ready()
            and not self.all_args.prediction_disable
        )
        if not use_prediction:
            return None, None, prediction_prev, False

        trajectory, visibility_mask = trajectory_buffer.get_trajectory()
        trajectory = trajectory.transpose(0, 1)
        visibility_mask = visibility_mask.transpose(0, 1)

        pred, uncertainty = self.trainer.policy.get_prediction(
            trajectory=trajectory,
            visibility_mask=visibility_mask,
            prediction_prev=prediction_prev,
            has_sample_dim=True,
        )

        pred = pred.detach().cpu()
        uncertainty = uncertainty.detach().cpu()
        prediction_prev = pred

        return pred, uncertainty, prediction_prev, True

    def _get_prediction_obs_shape(self, obs, share_obs, infos):
        """Infer the tensor shape used by the predictor for the current rollout."""
        if infos and "observation_global" in infos[0] and self.all_args.state_encoder:
            obs_global = np.array(infos[0]["observation_global"])
            return obs_global.shape[1:] if obs_global.ndim >= 2 else obs_global.shape
        if isinstance(obs, np.ndarray) and obs.dtype != object:
            return obs.shape[2:] if obs.ndim >= 3 else obs.shape[1:]
        if share_obs is not None:
            share_obs = np.array(share_obs)
            return share_obs.shape[2:] if share_obs.ndim >= 3 else share_obs.shape[1:]
        raise ValueError("Unable to infer predictor observation shape for render/eval.")

    def _format_policy_obs(self, obs):
        """Match the policy input layout used by the replay buffer."""
        if isinstance(obs, np.ndarray) and obs.dtype != object:
            return torch.from_numpy(einops.rearrange(obs, "t n ... -> (t n) ...")).float()

        if isinstance(obs, np.ndarray) and obs.dtype == object:
            if obs.ndim >= 3:
                return obs.reshape(-1, obs.shape[-1])
            if obs.ndim == 2 and obs.shape[-1] > 1:
                return obs.reshape(-1, obs.shape[-1])

        flat_obs = obs.reshape(-1)
        if flat_obs.shape[0] == 0:
            return flat_obs
        if isinstance(flat_obs[0], dict):
            return np.array([list(item.values()) for item in flat_obs], dtype=object)
        return flat_obs

    def _format_policy_global_obs(self, infos):
        """Build the per-agent global observation input expected by the state encoder."""
        if not infos or "observation_global" not in infos[0]:
            return None
        global_obs = np.array([info["observation_global"] for info in infos])
        return torch.from_numpy(global_obs.reshape(-1, *global_obs.shape[2:])).float()

    @torch.no_grad()
    def eval(self):
        log_root = zarr.open_group(self.all_args.eval_output_file, mode="a")
        if self.all_args.eval_series == "":
            log_exp = log_root.require_group(self.experiment_name)
        else:
            log_exp = log_root.require_group(self.all_args.eval_series).require_group(self.experiment_name)

        # Store configuration in the Zarr file as attributes.
        config_dict = vars(self.all_args)
        for key, value in config_dict.items():
            log_exp.attrs[key] = value

        eval_env = self.envs

        self.trainer.prep_rollout()

        eval_trajectory = TrajectoryBuffer(
            history_length=self.all_args.prediction_history_window,
            transition_includes_actions=self.all_args.prediction_history_include_actions,
        )

        prediction_prev = None  # For autoregression
        for i_episode in range(self.all_args.eval_episodes):
            obs, share_obs, available_actions, infos = eval_env.reset(return_info=True)
            obs_shape = self._get_prediction_obs_shape(obs, share_obs, infos)
            prediction_target, prediction_target_visibility = self._get_prediction_target(
                obs, infos, self.n_eval_rollout_threads, share_obs=share_obs
            )
            rnn_states = torch.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=torch.float32)
            masks = torch.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=torch.float32)
            self._seed_trajectory_buffer(eval_trajectory, obs, self.n_eval_rollout_threads, share_obs=share_obs)

            dones = False
            j = -1
            while not np.all(dones):
                j += 1

                aa = np.concatenate(available_actions)
                if np.any(aa == None):
                    aa = None

                prediction, uncertainty, prediction_prev, use_prediction = self._compute_predictions(
                    eval_trajectory,
                    prediction_prev,
                    self.n_eval_rollout_threads,
                    obs_shape,
                )
                prediction_error_mean = torch.zeros((), dtype=torch.float32)
                prediction_trajectory_error = None
                if use_prediction:
                    prediction_error_mean, prediction_trajectory_error = self._compute_prediction_errors(
                        prediction,
                        prediction_target,
                        prediction_target_visibility,
                        self.n_eval_rollout_threads,
                        obs_shape,
                    )

                obs_policy = self._format_policy_obs(obs)
                global_obs_policy = self._format_policy_global_obs(infos)
                uncertainty_now = torch.zeros((self.n_eval_rollout_threads, self.num_agents, *obs_shape), dtype=torch.float32)
                if use_prediction:
                    if global_obs_policy is not None and self.all_args.state_encoder:
                        global_obs_policy = einops.rearrange(prediction[:, -1], "(t n) ... -> (t n) ...", t=self.n_eval_rollout_threads, n=self.num_agents)
                    else:
                        obs_policy = einops.rearrange(prediction[:, -1], "(t n) ... -> (t n) ...", t=self.n_eval_rollout_threads, n=self.num_agents)
                    uncertainty_now = uncertainty[:, -1].reshape((self.n_eval_rollout_threads, self.num_agents, *obs_shape))

                actions, rnn_states = self.trainer.policy.act(
                    obs_policy,
                    rnn_states,
                    masks,
                    deterministic=False,
                    available_actions=aa,
                    global_obs=global_obs_policy
                )

                actions = actions.detach().cpu().reshape((self.n_eval_rollout_threads, self.num_agents, *actions.shape[1:]))
                rnn_states = rnn_states.detach()
                actions_env = [actions[idx, :, :].numpy() for idx in range(self.n_eval_rollout_threads)]

                obs, share_obs, eval_rewards, dones, infos, available_actions = eval_env.step(actions_env)
                prediction_target, prediction_target_visibility = self._get_prediction_target(
                    obs, infos, self.n_eval_rollout_threads, share_obs=share_obs
                )

                _, obs = self._build_viz_mask(obs, infos, actions, self.n_eval_rollout_threads)
                self._update_trajectory_buffer(eval_trajectory, obs, infos, actions, self.n_eval_rollout_threads, share_obs=share_obs)

                # Add prediction error to infos for logging.
                infos[0]["prediction_error_mean"] = prediction_error_mean.item() if use_prediction else 0.0
                infos[0]["prediction_trajectory_error_mean"] = prediction_trajectory_error.mean().item() if prediction_trajectory_error is not None else 0.0

                # Add uncertainty to infos for logging.
                infos[0]["prediction_uncertainty_mean"] = uncertainty_now.mean().item()
                if prediction_trajectory_error is not None:
                    uncertainty_gap = uncertainty_now - prediction_trajectory_error
                    infos[0]["prediction_uncertainty_gap_mean"] = uncertainty_gap.mean().item()
                    infos[0]["prediction_uncertainty_gap_min"] = uncertainty_gap.min().item()
                    infos[0]["prediction_uncertainty_gap_max"] = uncertainty_gap.max().item()
                else:
                    infos[0]["prediction_uncertainty_gap_mean"] = 0.0
                    infos[0]["prediction_uncertainty_gap_min"] = 0.0
                    infos[0]["prediction_uncertainty_gap_max"] = 0.0

                # Log information.
                for key in infos[0].keys():
                    if type(key) == str:
                        array = log_exp.require_dataset(
                            key,
                            shape=(
                                self.all_args.eval_episodes,
                                self.all_args.episode_length,
                                *np.array(infos[0][key]).shape
                            ),
                            dtype=np.float32,
                            fill_value=0.0,
                        )
                        array[i_episode, j] = np.array(infos[0][key])


    @torch.no_grad()
    def render(self, ipython_clear_output=True):        

        if ipython_clear_output:
            from IPython.display import clear_output

        render_env = self.envs

        self.trainer.prep_rollout()
                
        # Get shape of observation space.
        HISTORY_LENGTH = self.all_args.prediction_history_window
        render_trajectory = TrajectoryBuffer(
            history_length=self.all_args.prediction_history_window,
            transition_includes_actions=self.all_args.prediction_history_include_actions,
        )

        prediction_prev = None  # For autoregression
        for i_episode in range(self.all_args.render_episodes):
            # Reset the environment and get the initial observations.
            obs, share_obs, available_actions, infos = render_env.reset(return_info=True)
            obs_shape = self._get_prediction_obs_shape(obs, share_obs, infos)
            prediction_target, prediction_target_visibility = self._get_prediction_target(
                obs, infos, self.n_render_rollout_threads, share_obs=share_obs
            )
            rnn_states = torch.zeros((self.n_render_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=torch.float32)
            masks = torch.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=torch.float32)
            self._seed_trajectory_buffer(render_trajectory, obs, self.n_render_rollout_threads, share_obs=share_obs)

            if self.all_args.save_gifs:        
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            dones = False
            reward_total = 0.0
            while not np.all(dones):
                time_start = time.time()

                aa = np.concatenate(available_actions)
                if np.any(aa == None):
                    aa = None

                prediction, uncertainty, prediction_prev, use_prediction = self._compute_predictions(
                    render_trajectory,
                    prediction_prev,
                    self.n_render_rollout_threads,
                    obs_shape,
                )
                prediction_error_mean = torch.zeros((), dtype=torch.float32)
                prediction_trajectory_error = None
                if use_prediction:
                    prediction_error_mean, prediction_trajectory_error = self._compute_prediction_errors(
                        prediction,
                        prediction_target,
                        prediction_target_visibility,
                        self.n_render_rollout_threads,
                        obs_shape,
                    )

                if use_prediction:
                    for agentIdx in range(self.num_agents):
                        print(f"Mean Prediction Error ({agentIdx}): {prediction_error_mean:.2f}")
                        print(f"Mean Trajectory Prediction Error ({agentIdx}): {prediction_trajectory_error.mean():.2f}")

                obs_policy = self._format_policy_obs(obs)
                global_obs_policy = self._format_policy_global_obs(infos)
                prediction_render = None
                uncertainty_render = None
                if use_prediction:
                    if global_obs_policy is not None and self.all_args.state_encoder:
                        global_obs_policy = einops.rearrange(prediction[:, -1], "(t n) ... -> (t n) ...", t=self.n_render_rollout_threads, n=self.num_agents)
                    else:
                        obs_policy = einops.rearrange(prediction[:, -1], "(t n) ... -> (t n) ...", t=self.n_render_rollout_threads, n=self.num_agents)
                    prediction_render = prediction[:, :, :].reshape((self.n_render_rollout_threads, self.num_agents, HISTORY_LENGTH, *obs_shape))[0].transpose(0, 1)
                    uncertainty_render = uncertainty[:, :, :].reshape((self.n_render_rollout_threads, self.num_agents, HISTORY_LENGTH, *obs_shape))[0].transpose(0, 1)

                actions, rnn_states = self.trainer.policy.act(
                    obs_policy,
                    rnn_states,
                    masks,
                    deterministic=False,
                    available_actions=aa,
                    global_obs=global_obs_policy
                )

                # Prepare the actions for the environment.
                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                actions = actions.detach().cpu().reshape((self.n_render_rollout_threads, self.num_agents, *actions.shape[1:]))
                rnn_states = rnn_states.detach()
                actions_env = [actions[idx, :, :].numpy() for idx in range(self.n_render_rollout_threads)]

                if render_env.envs[0].env.step_count == 24:
                    print(f"ready")

                obs, share_obs, render_rewards, dones, infos, available_actions = render_env.step(actions_env)
                prediction_target, prediction_target_visibility = self._get_prediction_target(
                    obs, infos, self.n_render_rollout_threads, share_obs=share_obs
                )

                _, obs = self._build_viz_mask(obs, infos, actions, self.n_render_rollout_threads)
                self._update_trajectory_buffer(render_trajectory, obs, infos, actions, self.n_render_rollout_threads, share_obs=share_obs)

                reward_total += render_rewards[0].sum()

                time_stop = time.time()

                # Perform rendering.
                if ipython_clear_output:
                    clear_output(wait = True)
                spf = prediction_render if use_prediction else None
                render_env.envs[0].env.render(spf, history_length=HISTORY_LENGTH, uncertainty=uncertainty_render)

                # append frame
                if self.all_args.save_gifs:        
                    image = infos[0]["frame"]
                    frames.append(image)
                
                # Print the FPS information.
                print(f"Step {render_env.envs[0].env.step_count} - FPS: {1 / (time_stop - time_start):.2f} (excluding render) - Reward Total: {reward_total:.2f}")

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )
