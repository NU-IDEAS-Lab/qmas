from collections import defaultdict, deque
from collections.abc import Iterable
from itertools import chain
import os
import time

import imageio
import numpy as np
import torch
import wandb
import zarr
from tqdm.auto import tqdm
from collections import deque
import matplotlib.pyplot as plt

from onpolicy.utils.util import update_linear_schedule, get_shape_from_act_space, get_shape_from_obs_space
from onpolicy.runner.shared.base_runner import Runner

from onpolicy.utils.shared_buffer_torchrl import SharedReplayBuffer


class PettingzooRunner(Runner):
    def __init__(self, config):

        # The default restore functionality is broken. Disable it and do it ourselves.
        model_dir = config['all_args'].model_dir
        config['all_args'].model_dir = None

        super(PettingzooRunner, self).__init__(config)

        # Override the default buffer with our new TorchRL one.
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

        self.env_infos = defaultdict(list)
       
        # Perform restoration.
        config['all_args'].model_dir = model_dir
        self.model_dir = config['all_args'].model_dir
        if self.model_dir is not None:
            self.restore(self.model_dir)
       
    def run(self):
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in (progress_bar := tqdm(range(episodes))):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            # Reset the environment and perform warmup.
            self.warmup()

            # Set the delta steps to 1.
            delta_steps = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
            for step in range(self.episode_length):
                # Sample actions, collect values and probabilities.
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                
                # Take a step in the environment and get the results.
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                # Get the number of steps taken by each agent since the agent was last ready.
                delta_steps = np.array([info["deltaSteps"] for info in infos])

                # insert data into buffer
                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, delta_steps, available_actions
                self.insert(data)

            # Get certain stats.
            avg_episode_rewards = self.buffer.rewards.mean().item() * self.episode_length

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                
                train_infos["average_episode_rewards"] = avg_episode_rewards
                train_infos["fps"] = total_num_steps / (end - start)
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
        obs, share_obs, available_actions = self.envs.reset()

        

        # Get the shape of the action space.
        act_shape = get_shape_from_act_space(self.buffer.act_space)
        if isinstance(act_shape, Iterable):
            actions_shape = (self.n_rollout_threads, self.num_agents, *act_shape)
        else:
            actions_shape = (self.n_rollout_threads, self.num_agents, act_shape)
        
        # Get the shape of action log probabilities from the policy.
        action_log_prob_shape = (self.n_rollout_threads, self.num_agents, self.policy.actor.act.log_prob_dim)
        

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
            available_actions=available_actions
        )



    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        share_obs, obs, rnn_states, rnn_states_critic, masks, available_actions = self.buffer.compatibility_get_policy_input(step)

        values, action, action_log_prob, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            masks,
            available_actions=available_actions
        )

        # The value function predictions are made once over the entire share_obs.
        # However, we need to compare them with per-agent returns.
        # This is done by repeating the value predictions for each agent.
        values = values.detach().cpu().reshape((self.n_rollout_threads, 1, 1)).repeat(1, self.policy.args.num_agents, 1)

        actions = action.detach().cpu().reshape((self.n_rollout_threads, self.num_agents, *action.shape[1:]))
        action_log_probs = action_log_prob.detach().cpu().reshape((self.n_rollout_threads, self.num_agents, *action_log_prob.shape[1:]))
        rnn_states = rnn_states.detach().cpu().reshape((self.n_rollout_threads, self.num_agents, *rnn_states.shape[1:]))
        rnn_states_critic = rnn_states_critic.detach().cpu().reshape((self.n_rollout_threads, self.num_agents, *rnn_states_critic.shape[1:]))

        if actions.shape[-1] == 1:
            actions_env = [actions[idx, :, 0].numpy() for idx in range(self.n_rollout_threads)]
        else:
            actions_env = [actions[idx, :, :].numpy() for idx in range(self.n_rollout_threads)]
        

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, delta_steps, available_actions = data
        
        # update env_infos if done
        dones_env = np.all(dones, axis=-1)

        # Get visibility mask from infos.
        visibility_mask = None
        if "visibility_mask" in infos[0]:
            visibility_mask = np.array([info["visibility_mask"] for info in infos])

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
            visibility_mask=visibility_mask
        )


    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()

        share_obs, obs, rnn_states, rnn_states_critic, masks, available_actions = self.buffer.compatibility_get_policy_input(-1)

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
                    wandb.log({k: np.mean(v, axis=0)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)    

    # @torch.no_grad()
    def eval(self):
        log_root = zarr.open_group(self.all_args.eval_output_file, mode="a")
        log_exp = log_root.require_group(self.experiment_name)

        eval_env = self.envs

        # Get shape of observation and action spaces.
        obs_shape = get_shape_from_obs_space(self.buffer.obs_space)
        act_shape = get_shape_from_act_space(self.buffer.act_space)
        obs_size = np.prod(obs_shape)
        act_size = np.prod(act_shape)
        transition_size = obs_size + act_size

        # eval trajectory
        HISTORY_LENGTH = self.all_args.diffusion_horizon
        buffer = [deque(maxlen=HISTORY_LENGTH) for _ in range(self.num_agents)]
        prediction = torch.zeros((HISTORY_LENGTH, self.num_agents, *obs_shape), dtype=torch.float32)
        
        prediction_prev = None  # For autoregression
        for i_episode in range(self.all_args.eval_episodes):
            for i in range(self.num_agents):
                buffer[i].clear()

            # Reset the environment and get the initial observations.
            obs, share_obs, available_actions = eval_env.reset()
            rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for i in range(self.num_agents):
                transition = np.concatenate((np.zeros(act_size, dtype=np.float32), obs[0][i]), axis=0)
                buffer[i].append({
                    "transition": torch.from_numpy(transition).to(self.device),
                    "visibility_mask": torch.ones(transition.shape, dtype=torch.float32, device=self.device)
                })

            dones = False
            j = -1
            while not np.all(dones):
                j += 1
                self.trainer.prep_rollout()

                aa = np.concatenate(available_actions)
                if np.any(aa == None):
                    aa = None
                
                # Use the prediction from the diffuser if available.
                use_prediction = hasattr(self.policy, "diffuser") and len(buffer[0]) == HISTORY_LENGTH and not self.all_args.diffusion_disable
                if use_prediction:
                    for agentIdx in range(self.num_agents):
                        trajectory = torch.stack([t["transition"] for t in buffer[agentIdx]], dim=0)
                        visibility_mask = torch.stack([t["visibility_mask"] for t in buffer[agentIdx]], dim=0)

                        # Get the prediction from the diffuser.
                        pred = self.trainer.policy.get_prediction(
                            trajectory=trajectory,
                            visibility_mask=visibility_mask,
                            prediction_prev=prediction_prev
                        )

                        # Store as the previous prediction.
                        prediction_prev = pred.detach()

                        # Only take the first sample (n_samples is 1 anyway).
                        # Strip the action part of the prediction.
                        prediction[:, agentIdx, :] = pred[0, :, act_size:]
                else:
                    prediction.zero_()
                    prediction[-1] = torch.from_numpy(obs[0])
                
                actions, rnn_states = self.trainer.policy.act(
                    prediction[-1], # Use the final timestep of the prediction.
                    np.concatenate(rnn_states if isinstance(rnn_states, (list, tuple)) else [rnn_states]),
                    np.concatenate(masks),
                    deterministic=True,
                    available_actions=aa
                )

                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                actions = actions.detach().cpu().reshape((self.n_render_rollout_threads, self.num_agents, *actions.shape[1:]))
                # rnn_states = rnn_states.detach().cpu().reshape((self.n_render_rollout_threads, self.num_agents, *rnn_states.shape[1:]))
                rnn_states = rnn_states.detach().cpu().reshape((self.n_render_rollout_threads, *rnn_states.shape[1:]))

                actions_env = [actions[idx, :, :].numpy() for idx in range(self.n_render_rollout_threads)]

                # Take a step in the environment.
                obs, share_obs, eval_rewards, dones, infos, available_actions = eval_env.step(actions_env)
                viz_mask_obs = np.expand_dims(infos[0]["visibility_mask"], 0) if "visibility_mask" in infos[0] else np.ones_like(obs)
                viz_mask_actions = np.ones(actions.shape, dtype=np.float32)  # Assuming actions are fully visible.
                viz_mask = np.concatenate([viz_mask_actions, viz_mask_obs], axis=-1)
                for i in range(self.num_agents):
                    transition = np.concatenate((actions[0][i].float(), obs[0][i]), axis=0)
                    agent_viz_mask = viz_mask[0, i]
                    buffer[i].append({
                        "transition": torch.from_numpy(transition).to(self.device),
                        "visibility_mask": torch.from_numpy(agent_viz_mask).to(self.device)
                    })

                # Log information.
                keys = infos[0].keys()
                for key in keys:
                    if type(key) == str:
                        # Set up the Zarr array if it doesn't exist.
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
                        # self.env_infos[key] = [i[key] for i in infos]

    # @torch.no_grad()
    def render(self, ipython_clear_output=True):        

        if ipython_clear_output:
            from IPython.display import clear_output

        render_env = self.envs
                
        # Get shape of observation and action spaces.
        obs_shape = get_shape_from_obs_space(self.buffer.obs_space)
        act_shape = get_shape_from_act_space(self.buffer.act_space)
        obs_size = np.prod(obs_shape)
        act_size = np.prod(act_shape)
        transition_size = obs_size + act_size

        # eval trajectory
        HISTORY_LENGTH = self.all_args.diffusion_horizon
        buffer = [deque(maxlen=HISTORY_LENGTH) for _ in range(self.num_agents)]
        prediction = torch.zeros((HISTORY_LENGTH, self.num_agents, *obs_shape), dtype=torch.float32)
        
        prediction_prev = None  # For autoregression
        for i_episode in range(self.all_args.render_episodes):
            for i in range(self.num_agents):
                buffer[i].clear()

            # Reset the environment and get the initial observations.
            obs, share_obs, available_actions = render_env.reset()
            rnn_states = np.zeros((self.n_render_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for i in range(self.num_agents):
                transition = np.concatenate((np.zeros(act_size, dtype=np.float32), obs[0][i].flatten()), axis=0)
                buffer[i].append({
                    "transition": torch.from_numpy(transition).to(self.device),
                    "visibility_mask": torch.ones(transition.shape, dtype=torch.float32, device=self.device)
                })

            if self.all_args.save_gifs:        
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            dones = False
            while not np.all(dones):
                time_start = time.time()

                self.trainer.prep_rollout()

                aa = np.concatenate(available_actions)
                if np.any(aa == None):
                    aa = None
                
                # Use the prediction from the diffuser if available.
                use_prediction = hasattr(self.policy, "diffuser") and len(buffer[0]) == HISTORY_LENGTH and not self.all_args.diffusion_disable
                if use_prediction:
                    for agentIdx in range(self.num_agents):
                        trajectory = torch.stack([t["transition"] for t in buffer[agentIdx]], dim=0)
                        visibility_mask = torch.stack([t["visibility_mask"] for t in buffer[agentIdx]], dim=0)

                        # Get the prediction from the diffuser.
                        pred = self.trainer.policy.get_prediction(
                            trajectory=trajectory,
                            visibility_mask=visibility_mask,
                            prediction_prev=prediction_prev
                        )

                        # Store as the previous prediction.
                        prediction_prev = pred.detach()

                        # Only take the first sample (n_samples is 1 anyway).
                        # Strip the action part of the prediction.
                        prediction[:, agentIdx, :] = pred[0, :, act_size:]
                else:
                    prediction.zero_()
                    prediction[-1] = torch.from_numpy(obs[0])
                
                actions, rnn_states = self.trainer.policy.act(
                    prediction[-1], # Use the final timestep of the prediction.
                    np.concatenate(rnn_states if isinstance(rnn_states, (list, tuple)) else [rnn_states]),
                    np.concatenate(masks),
                    deterministic=True,
                    available_actions=aa
                )

                # Perform rendering.
                if ipython_clear_output:
                    clear_output(wait = True)
                spf = prediction if use_prediction else None
                render_env.envs[0].env.render(spf, history_length=HISTORY_LENGTH)

                # Prepare the actions for the environment.
                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                actions = actions.detach().cpu().reshape((self.n_render_rollout_threads, self.num_agents, *actions.shape[1:]))
                rnn_states = rnn_states.detach().cpu().reshape((self.n_render_rollout_threads, *rnn_states.shape[1:]))
                actions_env = [actions[idx, :, :].numpy() for idx in range(self.n_render_rollout_threads)]

                # Take a step in the environment and get the results.
                obs, share_obs, render_rewards, dones, infos, available_actions = render_env.step(actions_env)
                viz_mask_obs = np.expand_dims(infos[0]["visibility_mask"], 0) if "visibility_mask" in infos[0] else np.ones_like(obs)
                viz_mask_actions = np.ones(actions.shape, dtype=np.float32)  # Assuming actions are fully visible.
                viz_mask = np.concatenate([viz_mask_actions, viz_mask_obs.reshape(self.n_render_rollout_threads, self.num_agents, -1)], axis=-1)
                for i in range(self.num_agents):
                    transition = np.concatenate((actions[0][i].float(), obs[0][i].flatten()), axis=0)
                    agent_viz_mask = viz_mask[0, i]
                    buffer[i].append({
                        "transition": torch.from_numpy(transition).to(self.device),
                        "visibility_mask": torch.from_numpy(agent_viz_mask).to(self.device)
                    })

                time_stop = time.time()

                # append frame
                if self.all_args.save_gifs:        
                    image = infos[0]["frame"]
                    frames.append(image)
                
                # Print the FPS information.
                print(f"Step {render_env.envs[0].env.step_count} - FPS: {1 / (time_stop - time_start):.2f}, Time per step: {time_stop - time_start:.4f}s (excluding render)")

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )
            
            # time.sleep(3.0)s