from collections import defaultdict, deque
from collections.abc import Iterable
from itertools import chain
import os
import time

import imageio
import numpy as np
import torch
import wandb
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

            if episode == 0:
                print("\n\n\nTraining Progress:") #give some space for the progress bar
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

        # Initialize buffer.
        self.buffer.insert(
            share_obs=share_obs,
            obs=obs,
            rnn_states_actor=np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32),
            rnn_states_critic=np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32),
            actions=np.zeros(actions_shape, dtype=np.float32),
            action_log_probs=np.zeros(actions_shape, dtype=np.float32),

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
            visibility_mask = np.array(np.split(visibility_mask, self.n_rollout_threads))

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
                if self.use_wandb:
                    wandb.log({k: np.mean(v, axis=0)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)    

    @torch.no_grad()
    def eval(self, total_num_steps):
        # reset envs and init rnn and mask
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # init eval goals
        num_done = 0
        eval_goals = np.zeros(self.all_args.eval_episodes)
        eval_win_rates = np.zeros(self.all_args.eval_episodes)
        eval_steps = np.zeros(self.all_args.eval_episodes)
        step = 0
        quo = self.all_args.eval_episodes // self.n_eval_rollout_threads
        rem = self.all_args.eval_episodes % self.n_eval_rollout_threads
        done_episodes_per_thread = np.zeros(self.n_eval_rollout_threads, dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + quo
        eval_episodes_per_thread[:rem] += 1
        unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

        # loop until enough episodes
        while num_done < self.all_args.eval_episodes and step < self.episode_length:
            # get actions
            self.trainer.prep_rollout()

            # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                torch.concatenate(list(eval_obs)),
                torch.concatenate(list(eval_rnn_states)),
                torch.concatenate(list(eval_masks)),
                deterministic=self.all_args.eval_deterministic
            )
            
            # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
            eval_actions = eval_actions.detach().cpu().reshape((self.n_eval_rollout_threads, self.num_agents, *eval_actions.shape[1:]))
            eval_rnn_states = eval_rnn_states.detach().cpu().reshape((self.n_eval_rollout_threads, self.num_agents, *eval_rnn_states.shape[1:]))

            eval_actions_env = [eval_actions[idx, :, 0].numpy() for idx in range(self.n_eval_rollout_threads)]

            # step
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

            # update goals if done
            eval_dones_env = np.all(eval_dones, axis=-1)
            eval_dones_unfinished_env = eval_dones_env[unfinished_thread]
            if np.any(eval_dones_unfinished_env):
                for idx_env in range(self.n_eval_rollout_threads):
                    if unfinished_thread[idx_env] and eval_dones_env[idx_env]:
                        eval_goals[num_done] = eval_infos[idx_env]["score_reward"]
                        eval_win_rates[num_done] = 1 if eval_infos[idx_env]["score_reward"] > 0 else 0
                        eval_steps[num_done] = eval_infos[idx_env]["max_steps"] - eval_infos[idx_env]["steps_left"]
                        # print("episode {:>2d} done by env {:>2d}: {}".format(num_done, idx_env, eval_infos[idx_env]["score_reward"]))
                        num_done += 1
                        done_episodes_per_thread[idx_env] += 1
            unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

            # reset rnn and masks for done envs
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            step += 1

        # get expected goal
        eval_goal = np.mean(eval_goals)
        eval_win_rate = np.mean(eval_win_rates)
        eval_step = np.mean(eval_steps)
    
        # log and print
        print("eval expected goal is {}.".format(eval_goal))
        if self.use_wandb:
            wandb.log({"eval_goal": eval_goal}, step=total_num_steps)
            wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
            wandb.log({"eval_step": eval_step}, step=total_num_steps)
        else:
            self.writter.add_scalars("eval_goal", {"expected_goal": eval_goal}, total_num_steps)
            self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
            self.writter.add_scalars("eval_step", {"expected_step": eval_step}, total_num_steps)

    # @torch.no_grad()
    def render(self, ipython_clear_output=True):        

        if ipython_clear_output:
            from IPython.display import clear_output

        # reset envs and init rnn and mask
        render_env = self.envs
                
        # Get shape of observation and action spaces.
        obs_shape = get_shape_from_obs_space(self.buffer.obs_space)
        act_shape = get_shape_from_act_space(self.buffer.act_space)
        obs_size = np.prod(obs_shape)
        act_size = np.prod(act_shape)
        transition_size = obs_size + act_size

        # eval trajectory
        HISTORY_LENGTH = self.all_args.diffusion_horizon
        trajectory = [deque(maxlen=HISTORY_LENGTH) for _ in range(self.num_agents)]
        state_pred = torch.zeros((self.num_agents, *obs_shape), dtype=torch.float32)
        state_pred_full = torch.zeros((HISTORY_LENGTH, self.num_agents, *obs_shape), dtype=torch.float32)
        
        for i_episode in range(self.all_args.render_episodes):
            for i in range(self.num_agents):
                trajectory[i].clear()

            # Reset the environment and get the initial observations.
            obs, share_obs, available_actions = render_env.reset()
            rnn_states = np.zeros((self.n_render_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for i in range(self.num_agents):
                transition = np.concatenate((np.zeros(act_size, dtype=np.float32), obs[0][i]), axis=0)
                trajectory[i].append(torch.from_numpy(transition))

            if self.all_args.save_gifs:        
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            dones = False
            while not np.all(dones):
                self.trainer.prep_rollout()

                aa = np.concatenate(available_actions)
                if np.any(aa == None):
                    aa = None

                for agentIdx in range(self.num_agents):
                    if len(trajectory[agentIdx]) == HISTORY_LENGTH: # and render_env.envs[0].env.step_count == HISTORY_LENGTH - 1:
                        
                        trajectory_tensor = torch.stack(list(trajectory[agentIdx]), dim=0)
                        
                        # Set up conditions (prior knowledge).
                        condition = np.zeros((1, HISTORY_LENGTH, transition_size), dtype=np.float32)
                        condition_mask = np.zeros((1, HISTORY_LENGTH, transition_size), dtype=np.float32)
                        for i in range(len(trajectory_tensor)):
                            # if (np.random.rand() > 0.5 and i<7) or i == 0: 
                            #  # Ensure at least t=0 is always conditioned
                            # if i < 4:
                            # if True:
                            if i == 0 or np.random.rand() > 0.8:
                            # if i % 2 == 0 or i == 1:
                                condition[0, i] = trajectory_tensor[i]
                                condition_mask[0, i] = 1.0
                        condition = torch.from_numpy(condition).to(self.device)
                        condition_mask = torch.from_numpy(condition_mask).to(self.device)

                        # The prior and fix_mask represent the known data and are applied as described by Janner et al.
                        # We set the fix_mask manually here as a workaround for CleanDiffuser not taking it as an input.
                        self.policy.diffuser.fix_mask = torch.nn.Parameter(condition_mask, requires_grad=False)
                        pred, log = self.policy.diffuser.sample(
                            prior=condition,

                            solver="ddpm",
                            n_samples=1,
                            sample_steps = 5,

                            # The condition_cg and condition_cg_mask represent the known data and are used for the guide function.
                            condition_cg=condition,
                            condition_cg_mask=condition_mask,
                            w_cg=0.1,
                            w_cfg=0.0
                        )
                        
                        # Strip the action part of the prediction.
                        state_pred[agentIdx] = pred[0, -1, act_size:]
                        state_pred_full[:, agentIdx, :] = pred[0, :, act_size:]
                    else:
                        state_pred[agentIdx] = torch.from_numpy(obs[0, agentIdx])
                        state_pred_full[:, agentIdx, :] = torch.zeros((HISTORY_LENGTH, *obs_shape), dtype=torch.float32)
                
                actions, rnn_states = self.trainer.policy.act(
                    state_pred,
                    # np.concatenate(rnn_states)
                    np.concatenate(rnn_states if isinstance(rnn_states, (list, tuple)) else [rnn_states]),
                    np.concatenate(masks),
                    deterministic=True,
                    available_actions=aa
                )

                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                actions = actions.detach().cpu().reshape((self.n_render_rollout_threads, self.num_agents, *actions.shape[1:]))
                # rnn_states = rnn_states.detach().cpu().reshape((self.n_render_rollout_threads, self.num_agents, *rnn_states.shape[1:]))
                rnn_states = rnn_states.detach().cpu().reshape((self.n_render_rollout_threads, *rnn_states.shape[1:]))

                actions_env = [actions[idx, :, :] for idx in range(self.n_render_rollout_threads)]

                # Take a step in the environment and get the results.
                obs, share_obs, render_rewards, dones, infos, available_actions = render_env.step(actions_env)
                if "visibility_mask" in infos[0]:
                    print("Visibility Mask:")
                    for i, info in enumerate(infos):
                        print(f"Agent {i}: {info['visibility_mask']}")
                # obs_traj.append(torch.from_numpy(share_obs[0]))
                for i in range(self.num_agents):
                    transition = np.concatenate((actions[0][i], obs[0][i]), axis=0)
                    trajectory[i].append(torch.from_numpy(transition))
                                
                if not np.all(dones):
                    if ipython_clear_output:
                        clear_output(wait = True)
                    # render_env.envs[0].env.render(state_pred_full, history_length=HISTORY_LENGTH)
                    render_env.envs[0].env.render()
                    # render_env.envs[0].env.render()

                # append frame
                if self.all_args.save_gifs:        
                    image = infos[0]["frame"]
                    frames.append(image)

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )
            
            # time.sleep(3.0)
