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

        # Per-log-window diagnostic accumulators for the predictor.
        # All counters here reset together inside `_flush_pred_diagnostics` so that
        # each value reported in train_infos describes the latest log window only.
        self._reset_pred_diagnostics()

        self.env_infos = defaultdict(list)
               
        if self.all_args.torch_compile:
            self.train_compiled = torch.compile(self.train, fullgraph=False, mode="reduce-overhead")

    def _reset_pred_diagnostics(self):
        """Reset all per-log-window predictor diagnostic accumulators."""
        # #1 buffer writeback check, #2 predictor firing, #3 prediction quality.
        self._pred_collect_calls = 0
        self._pred_fires = 0
        self._pred_writeback_checks = 0
        self._pred_writeback_failures = 0
        self._pred_quality_abs_err_sum = 0.0
        self._pred_quality_target_abs_sum = 0.0
        self._pred_quality_count = 0
        self._pred_quality_visible_err_sum = 0.0

    def _flush_pred_diagnostics(self, train_infos):
        """Merge predictor diagnostic counters into train_infos and reset them.

        Values describe the log window since the last call. Empty/zero windows
        report NaN for ratios/MAEs so the chart shows a gap rather than a misleading 0.
        """
        calls = self._pred_collect_calls
        fires = self._pred_fires
        wb_checks = self._pred_writeback_checks
        wb_fails = self._pred_writeback_failures
        q_count = self._pred_quality_count

        train_infos["pred_collect_calls"] = calls
        train_infos["pred_fires"] = fires
        train_infos["pred_fire_ratio"] = (fires / calls) if calls > 0 else float("nan")
        train_infos["pred_writeback_checks"] = wb_checks
        train_infos["pred_writeback_failures"] = wb_fails

        if q_count > 0:
            masked_mae = self._pred_quality_abs_err_sum / q_count
            target_mag = self._pred_quality_target_abs_sum / q_count
            train_infos["pred_masked_mae"] = masked_mae
            train_infos["pred_masked_target_mag"] = target_mag
            train_infos["pred_masked_norm_err"] = masked_mae / max(target_mag, 1e-9)
            train_infos["pred_visible_mae"] = self._pred_quality_visible_err_sum / q_count
        else:
            train_infos["pred_masked_mae"] = float("nan")
            train_infos["pred_masked_target_mag"] = float("nan")
            train_infos["pred_masked_norm_err"] = float("nan")
            train_infos["pred_visible_mae"] = float("nan")

        self._reset_pred_diagnostics()

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
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step, episode=episode, episodes=episodes)
                
                # Take a step in the environment and get the results.
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                # Get the number of steps taken by each agent since the agent was last ready.
                delta_steps = np.array([info["deltaSteps"] for info in infos])

                # insert data into buffer
                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, delta_steps, available_actions
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
                self.save(episode, quiet=True)
                last_save_step = total_num_steps

            # log information
            if total_num_steps % self.log_interval == 0:

                train_infos["average_episode_rewards"] = avg_episode_rewards
                train_infos["fps"] = self.episode_length * self.n_rollout_threads / (end - start_episode)
                # Predictor diagnostics (firing count, buffer writeback verification,
                # prediction quality at masked vs visible positions) — reset per window.
                self._flush_pred_diagnostics(train_infos)
                print(
                    f"[DIAG] episode={episode} "
                    f"pred_fires={train_infos['pred_fires']}/{train_infos['pred_collect_calls']} "
                    f"(ratio={train_infos['pred_fire_ratio']:.4f}), "
                    f"writeback_failures={train_infos['pred_writeback_failures']}/{train_infos['pred_writeback_checks']}, "
                    f"masked_mae={train_infos['pred_masked_mae']:.4g} "
                    f"(target_mag={train_infos['pred_masked_target_mag']:.4g}, "
                    f"norm_err={train_infos['pred_masked_norm_err']:.3f}), "
                    f"visible_mae={train_infos['pred_visible_mae']:.4g}"
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = defaultdict(list)

            progress_bar.set_postfix({
                # "exp": self.experiment_name,
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

        # Use the visibility mask from infos, falling back to all-ones if not provided.
        if "visibility_mask" in infos[0]:
            visibility_mask = np.array([info["visibility_mask"] for info in infos], dtype=np.float32)
        else:
            visibility_mask = np.ones_like(obs, dtype=np.float32)
        visibility_mask_global = None
        if "visibility_mask_global" in infos[0]:
            visibility_mask_global = np.array([info["visibility_mask_global"] for info in infos], dtype=np.float32)

        observation_global = None
        if "observation_global" in infos[0] and not self.all_args.disable_observation_global:
            observation_global = np.array([info["observation_global"] for info in infos])

        # Capture raw (pre-UQ-padding) versions for seeding the trajectory buffer below.
        obs_raw = obs
        observation_global_raw = observation_global

        # If UQ will be appended during rollout, pre-pad with zeros here so the
        # buffer slot has the correct shape (2× obs_dim) from the start.
        # collect() will overwrite this with [pred | uncertainty] on the first step.
        obs_full = None
        if self.all_args.prediction_uq_injection_method == "append" and not self.all_args.prediction_disable:
            if obs.dtype == object or observation_global is not None:
                # Object-type obs or global obs available: UQ is appended to observation_global
                # (matches collect() which writes to global_obs whenever global_obs is not None).
                if observation_global is None:
                    raise ValueError("Must have observation_global in infos to use object-type observations (e.g. pyg graph observations). Check that environment is providing this information.")
                observation_global = np.concatenate(
                    [observation_global, np.zeros_like(observation_global)], axis=-1
                )
            else:
                # Dense envs with no global obs: UQ is appended to obs itself. Pre-pad obs with
                # zeros so the buffer slot has the correct shape before collect() overwrites it.
                obs_full = obs
                obs = np.concatenate([obs, np.zeros_like(obs)], axis=-1)

        # Apply visibility mask to global obs before the initial insertion.
        if self.all_args.observation_mask and observation_global is not None and visibility_mask_global is not None:
            mask = visibility_mask_global
            if mask.shape[-1] < observation_global.shape[-1]:
                pad = np.ones((*mask.shape[:-1], observation_global.shape[-1] - mask.shape[-1]), dtype=np.float32)
                mask = np.concatenate([mask, pad], axis=-1)
            observation_global = observation_global * mask

        # Initialize buffer.
        self.buffer.insert(
            share_obs=share_obs,
            obs=obs,
            obs_full=obs_full,
            rnn_states_actor=np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32),
            rnn_states_critic=np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.critic_hidden_size), dtype=np.float32),
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
            visibility_mask=visibility_mask
        )

        # Reset and seed training trajectory buffer with the initial observations.
        # Mirrors _apply_masks_and_update_trajectory but uses zero actions and the
        # raw (pre-UQ-padding) obs so the predictor sees the correct observation size.
        self.train_prediction_prev = None
        self.train_trajectory.reset()

        if observation_global_raw is not None:
            traj_obs = observation_global_raw
            traj_visibility_mask = (
                visibility_mask_global
                if visibility_mask_global is not None
                else np.ones_like(traj_obs, dtype=np.float32)
            )
        elif obs_raw.dtype == object:
            raise ValueError(
                "Must have observation_global in infos to use object-type observations "
                "(e.g. pyg graph observations). Check that environment is providing this information."
            )
        else:
            traj_obs = obs_raw
            traj_visibility_mask = visibility_mask

        if self.all_args.observation_mask:
            traj_obs = traj_obs * traj_visibility_mask

        traj_obs = einops.rearrange(traj_obs, "t n ... -> (t n) ...")
        traj_visibility_mask = einops.rearrange(traj_visibility_mask, "t n ... -> (t n) ...")

        act_size = int(np.prod(get_shape_from_act_space(self.buffer.act_space)))
        traj_actions = torch.zeros(
            (self.n_rollout_threads * self.num_agents, act_size),
            dtype=torch.float32,
            device=self.device,
        )

        self.train_trajectory.add(
            obs=torch.from_numpy(traj_obs).to(self.device),
            visibility_mask=torch.from_numpy(traj_visibility_mask).to(self.device),
            action=traj_actions,
        )


    @torch.no_grad()
    def collect(self, step, episode=None, episodes=None):
        share_obs, obs, global_obs, rnn_states, rnn_states_critic, masks, available_actions = self.buffer.compatibility_get_policy_input(step)

        uncertainty_now = None
        pred_now = None
        update_obs = False
        update_global_obs = False

        # If a predictor is available, replace obs with predictions from the trajectory buffers.
        # Use episode-based fraction to stay consistent with the training condition in algorithm.py.
        episode_fraction = (episode / episodes) if (episode is not None and episodes) else 0.0
        use_prediction = (
            hasattr(self.policy, "predictors")
            and not self.all_args.prediction_disable
            and self.all_args.episode_fraction_start_prediction <= episode_fraction
            and self.train_trajectory.ready()
            and (self.all_args.prediction_during_training or self.all_args.prediction_uq_injection_method != "none")
        )
        self._pred_collect_calls += 1
        if use_prediction:
            self._pred_fires += 1
        if use_prediction:
            # Get the prediction and the uncertainty.
            trajectory, visibility_mask = self.train_trajectory.get_trajectory()
            trajectory = trajectory.transpose(0, 1)
            visibility_mask = visibility_mask.transpose(0, 1)
            pred, uncertainty, member_preds = self.trainer.policy.get_prediction(
                trajectory=trajectory,
                visibility_mask=visibility_mask,
                prediction_prev=self.train_prediction_prev,
                has_sample_dim=True,
                return_member_preds=True
            )
            pred = pred.detach().cpu()
            self.train_prediction_prev = member_preds.detach().cpu()
            uncertainty = uncertainty.detach().cpu()

            # Get the predicted current state.
            pred_now = pred[:, -1]
            uncertainty_now = uncertainty[:, -1]

            # Apply any ex-environment comms merge (no-op when disabled).
            pred_now = self._apply_ex_env_comms_merge(
                pred_now,
                uncertainty_now,
                self.env_infos.get("received_comms", []),
                self.train_prediction_prev,
            )

            # When the trajectory includes actions (prepended), strip the action prefix so
            # that pred_now and uncertainty_now contain only the obs dimensions.
            if self.all_args.prediction_history_include_actions:
                act_size = int(np.prod(get_shape_from_act_space(self.buffer.act_space)))
                pred_now = pred_now[..., act_size:]
                uncertainty_now = uncertainty_now[..., act_size:]

            # Replace the observation with the prediction.
            if self.all_args.prediction_during_training:
                obs_dim = pred_now.shape[-1]
                if (self.all_args.state_encoder and global_obs is not None) or obs.dtype == object:
                    global_obs[..., :obs_dim] = pred_now
                    update_global_obs = True
                else:
                    obs[..., :obs_dim] = pred_now
                    update_obs = True
            
            # Inject uncertainty into the observation.
            if self.all_args.prediction_uq_injection_method == "append":
                obs_dim = uncertainty_now.shape[-1]
                if global_obs is not None:
                    global_obs[..., obs_dim:] = uncertainty_now
                    update_global_obs = True
                elif obs.dtype == object:
                    raise ValueError("Must have observation_global in infos to use object-type observations (e.g. pyg graph observations). Check that environment is providing this information.")
                else:
                    obs[..., obs_dim:] = uncertainty_now
                    update_obs = True

        # Call the policy.
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

        # The environment expects numpy-based actions.
        actions_env = [actions[idx, :, :].numpy() for idx in range(self.n_rollout_threads)]
        
        updates = {}
        if update_obs:
            updates["obs"] = obs.reshape((self.n_rollout_threads, self.num_agents, *obs.shape[1:]))
        if update_global_obs:
            updates["observation_global"] = global_obs.reshape((self.n_rollout_threads, self.num_agents, *global_obs.shape[1:]))
        if updates:
            self.buffer.update_step(step, **updates)

            # DIAGNOSTIC #1: confirm pred_now actually landed in the buffer that the
            # trainer will read. Sample a few entries and compare to what we wrote.
            # If this assertion fires, the storage write is silently dropped.
            if use_prediction and update_obs:
                self._pred_writeback_checks += 1
                check_share_obs, check_obs, check_global_obs, *_ = self.buffer.compatibility_get_policy_input(step)
                # Compare on the slice we actually overwrote.
                expected = obs[..., :pred_now.shape[-1]]
                got = check_obs[..., :pred_now.shape[-1]]
                if not torch.allclose(
                    got.float() if isinstance(got, torch.Tensor) else torch.as_tensor(got, dtype=torch.float32),
                    expected.float() if isinstance(expected, torch.Tensor) else torch.as_tensor(expected, dtype=torch.float32),
                    atol=1e-5, rtol=1e-4,
                ):
                    self._pred_writeback_failures += 1
                    diff = (got - expected).abs()
                    print(
                        f"[DIAG] update_step did NOT stick at step={step}: "
                        f"max|got-expected|={diff.max().item():.4g}, "
                        f"mean|got-expected|={diff.mean().item():.4g}"
                    )

            # DIAGNOSTIC #3: how good is pred_now at the masked positions?
            # Compares pred_now against the ground-truth `obs_full` stored in the
            # buffer (the un-masked obs from this step). Splits the error into:
            #   - masked positions (fix_mask=0): predictor's job — should shrink with training.
            #   - visible positions (fix_mask=1): preserved by diffuser fix_mask — should be ~0.
            if use_prediction and update_obs and pred_now is not None:
                try:
                    sample = self.buffer[step]
                    if "obs_full" in sample:
                        obs_full_step = sample["obs_full"].flatten(0, 1).float()  # (B, D_full)
                        # Slice to the predictor's output dim (handles UQ padding).
                        D = pred_now.shape[-1]
                        obs_full_step = obs_full_step[..., :D].cpu()
                        pred_cpu = pred_now.float().cpu()

                        # Per-step visibility mask from the trajectory we just used.
                        # When actions are prepended to transitions, viz has shape
                        # (B, D_action + D_obs); strip the action prefix so the mask
                        # aligns with pred_now and obs_full (both D_obs wide). Without
                        # this, every column gets relabeled by D_action and the
                        # masked/visible reports become meaningless.
                        viz_now = visibility_mask[:, -1].float().cpu()  # 1=visible, 0=masked
                        if self.all_args.prediction_history_include_actions:
                            act_size = int(np.prod(get_shape_from_act_space(self.buffer.act_space)))
                            if viz_now.shape[-1] >= D + act_size:
                                viz_now = viz_now[..., act_size:act_size + D]
                        if viz_now.shape[-1] > D:
                            viz_now = viz_now[..., :D]
                        elif viz_now.shape[-1] < D:
                            # Pad with ones to match (shouldn't normally happen).
                            pad = torch.ones((*viz_now.shape[:-1], D - viz_now.shape[-1]))
                            viz_now = torch.cat([viz_now, pad], dim=-1)

                        masked = (viz_now < 0.5)   # positions the predictor must fill
                        visible = ~masked

                        diff = (pred_cpu - obs_full_step).abs()
                        if masked.any():
                            self._pred_quality_abs_err_sum += diff[masked].sum().item()
                            self._pred_quality_target_abs_sum += obs_full_step[masked].abs().sum().item()
                            self._pred_quality_count += int(masked.sum().item())
                        if visible.any():
                            self._pred_quality_visible_err_sum += diff[visible].sum().item()
                except Exception as e:
                    # Don't let diagnostics break training.
                    print(f"[DIAG] prediction-quality check failed: {e}")

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, delta_steps, available_actions = data
        
        # update env_infos if done
        dones_env = np.all(dones, axis=-1)

        # Get visibility mask from infos.
        visibility_mask = None
        if "visibility_mask" in infos[0]:
            visibility_mask = np.array([info["visibility_mask"] for info in infos])
        visibility_mask_global = None
        if "visibility_mask_global" in infos[0]:
            visibility_mask_global = np.array([info["visibility_mask_global"] for info in infos])
        
        obs_full = None
        obs_raw = obs  # unpadded, for trajectory buffer
        if "observation_global" in infos[0] and not self.all_args.disable_observation_global:
            observation_global = np.array([info["observation_global"] for info in infos])
            observation_global_raw = observation_global  # unpadded, for trajectory buffer
            # The buffer was initialized with zero-padded obs (2× size) in warmup().
            # Pad the raw env obs here to match, so the storage set doesn't fail on shape.
            if self.all_args.prediction_uq_injection_method == "append" and (obs.dtype == object or self.all_args.state_encoder):
                observation_global = np.concatenate(
                    [observation_global, np.zeros_like(observation_global)], axis=-1
                )
            # Apply visibility mask to global obs before storing in the buffer.
            if self.all_args.observation_mask and visibility_mask_global is not None:
                # Pad the mask with ones to match the (possibly UQ-padded) observation shape.
                mask = visibility_mask_global
                if mask.shape[-1] < observation_global.shape[-1]:
                    pad = np.ones((*mask.shape[:-1], observation_global.shape[-1] - mask.shape[-1]), dtype=np.float32)
                    mask = np.concatenate([mask, pad], axis=-1)
                observation_global = observation_global * mask
        else:
            observation_global = None
            observation_global_raw = None
            # Dense envs without state_encoder: pad obs with zeros for the same reason.
            if obs.dtype != object and not self.all_args.state_encoder and self.all_args.prediction_uq_injection_method == "append":
                obs_full = obs
                obs = np.concatenate([obs, np.zeros_like(obs)], axis=-1)

        # Get extra state information from infos.
        state_visibility_mask = None
        if "state_visibility_mask" in infos[0] and not self.all_args.disable_observation_global:
            state_visibility_mask = np.array([info["state_visibility_mask"] for info in infos])

        # Add information to the logger.
        keys = infos[0].keys()
        for key in keys:
            if type(key) == str:
                self.env_infos[key] = [i[key] for i in infos]

        # Calculate masks.
        masks = torch.ones((self.n_rollout_threads, self.num_agents, 1))
        for i in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                if dones[i, agent_id]:
                    rnn_states[i][agent_id] = torch.zeros((self.recurrent_N, self.hidden_size))
                    rnn_states_critic[i][agent_id] = torch.zeros((self.recurrent_N, self.critic_hidden_size))
                    masks[i, agent_id] = torch.zeros(1)

        self.buffer.insert(
            share_obs=share_obs,
            obs=obs,
            obs_full=obs_full,
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
        )

        # Update the trajectory buffer with the new transition.
        # Use raw (unpadded) obs/global_obs so the predictor sees the correct observation size.
        self._apply_masks_and_update_trajectory(
            trajectory_buffer=self.train_trajectory,
            obs=obs_raw,
            infos=infos,
            actions=actions,
            n_threads=self.n_rollout_threads,
            global_obs=observation_global_raw
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

    def _apply_masks_and_update_trajectory(self, trajectory_buffer, obs, infos, actions, n_threads, global_obs=None):
        """Apply visibility masks to observations, then add the masked transition to the trajectory buffer.

        Both local obs (when not object-dtype) and global_obs are masked when
        observation_mask is set. The trajectory buffer receives whichever observation
        type is appropriate (global_obs when present, otherwise obs), along with its
        matching visibility mask. Returns masked (obs, global_obs).
        """
        # Build visibility masks from infos.
        viz_mask_local = np.array([info["visibility_mask"] for info in infos]) if "visibility_mask" in infos[0] else np.ones_like(obs)
        viz_mask_global = np.array([info["visibility_mask_global"] for info in infos]) if "visibility_mask_global" in infos[0] else None

        # Perform observation masking.
        if self.all_args.observation_mask:
            if obs.dtype != object:
                obs = obs * viz_mask_local
            if global_obs is not None and viz_mask_global is not None:
                global_obs = global_obs * viz_mask_global

        # Select which obs and visibility mask go into the trajectory buffer.
        if global_obs is not None:
            traj_obs = global_obs
            traj_visibility_mask = viz_mask_global
        elif obs.dtype == object:
            raise ValueError("Must have observation_global in infos to use object-type observations (e.g. pyg graph observations). Check that environment is providing this information.")
        else:
            traj_obs = obs
            traj_visibility_mask = viz_mask_local
        assert traj_visibility_mask is not None, "Must have some form of visibility mask in infos to update trajectory buffer. Check that environment is providing this information."

        # Reshape to match expected input shape of trajectory buffer (T*N, ...).
        traj_obs = einops.rearrange(traj_obs, "t n ... -> (t n) ...")
        traj_visibility_mask = einops.rearrange(traj_visibility_mask, "t n ... -> (t n) ...")
        traj_actions = einops.rearrange(actions.detach().cpu(), "t n ... -> (t n) ...")

        # Add to the buffer.
        trajectory_buffer.add(
            obs=torch.from_numpy(traj_obs).to(self.device),
            visibility_mask=torch.from_numpy(traj_visibility_mask).to(self.device),
            action=traj_actions.to(self.device),
        )

        return obs, global_obs

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

    def _compute_predictions(self, trajectory_buffer, prediction_prev, n_threads, obs_shape, ground_truth_obs=None):
        use_prediction = (
            hasattr(self.policy, "predictors")
            and trajectory_buffer.ready()
            and not self.all_args.prediction_disable
            and (self.all_args.prediction_during_training or self.all_args.prediction_uq_injection_method != "none")
        )
        if not use_prediction:
            return None, None, prediction_prev, None, False

        trajectory, visibility_mask = trajectory_buffer.get_trajectory()
        trajectory = trajectory.transpose(0, 1)
        visibility_mask = visibility_mask.transpose(0, 1)

        pred, uncertainty, member_preds = self.trainer.policy.get_prediction(
            trajectory=trajectory,
            visibility_mask=visibility_mask,
            prediction_prev=prediction_prev,
            has_sample_dim=True,
            return_member_preds=True,
        )

        pred = pred.detach().cpu()
        uncertainty = uncertainty.detach().cpu()
        prediction_prev = member_preds.detach().cpu()

        pred_last = pred[:, -1]
        uncertainty_last = uncertainty[:, -1]
        if self.all_args.prediction_history_include_actions:
            act_size = int(np.prod(get_shape_from_act_space(self.buffer.act_space)))
            pred_last = pred_last[..., act_size:]
            uncertainty_last = uncertainty_last[..., act_size:]
        pred_now = pred_last.reshape((n_threads, self.num_agents, *obs_shape))
        uncertainty_now = uncertainty_last.reshape((n_threads, self.num_agents, *obs_shape))

        if ground_truth_obs is not None:
            # Use the unmasked ground truth (provided by the caller from infos before masking).
            # This allows measuring prediction error for unobserved positions too.
            ground_truth_now = torch.from_numpy(ground_truth_obs).float().reshape((n_threads, self.num_agents, *obs_shape))
            prediction_error = torch.abs(pred_now - ground_truth_now)
        else:
            # Fall back to the masked trajectory entry. Unobserved entries are zeroed, so
            # only include positions where ground truth was actually observed.
            transition_now = trajectory[:, -1].detach().cpu()
            if self.all_args.prediction_history_include_actions:
                act_size = int(np.prod(get_shape_from_act_space(self.buffer.act_space)))
                transition_now = transition_now[:, act_size:]
            transition_now = transition_now.reshape((n_threads, self.num_agents, *obs_shape))
            visibility_mask_now = visibility_mask[:, -1].detach().cpu().reshape((n_threads, self.num_agents, *obs_shape))
            prediction_error = torch.abs(pred_now - transition_now) * visibility_mask_now

        return pred, uncertainty, prediction_prev, prediction_error, True

    def _apply_ex_env_comms_merge(self, pred_now, uncertainty_now, received_comms_per_thread, prediction_prev):
        """Apply the configured ex-environment comms merge to ``pred_now`` and persist into autoregression.

        For each receiver agent (per ``received_comms_per_thread``), replace its prediction with the
        per-element values from the (within-thread) agent with the lowest uncertainty. Senders = all agents
        in the thread (including the receiver itself). Visibility is disregarded — assumes agents send
        their full prediction + uncertainty.

        Args:
            pred_now: ``(n_threads * n_agents, D)`` — per-agent prediction at the current step.
            uncertainty_now: ``(n_threads * n_agents, D)`` — per-agent uncertainty at the current step.
            received_comms_per_thread: iterable of length ``n_threads`` with each element either a
                per-thread bool (whole thread received) or a per-agent array of length ``n_agents``.
            prediction_prev: ``(n_ensemble, n_threads * n_agents, T, D)`` autoregression source.
                Receiver rows of the last timestep are overwritten in place with the merged values.

        Shapes are reshaped to ``(n_threads, n_agents, D)`` so argmin stays within each thread — mixing
        across threads is meaningless because they're independent rollouts.

        Timing: the caller is responsible for passing flags that describe the same env transition as
        ``pred_now``'s target obs (i.e., the most-recent step seen by the predictor). In ``collect`` this
        is ``self.env_infos`` (populated by the prior iteration's ``insert()``); in ``eval``/``render`` it's
        the cached previous-iter ``infos`` list.

        Persistence: writing back at receiver rows lets the next predictor call autoregress on the
        communicated info. Non-receivers retain per-ensemble diversity. Requires
        ``diffusion_autoregression_steps > 0`` (or ``prediction_warm_start`` for a softer effect) to
        actually carry forward — ideally with ``diffusion_autoregression_steps = prediction_history_window - 1``.

        Returns ``pred_now`` (possibly modified). No-op when the mode is disabled or no receivers fired.
        """
        if self.all_args.ex_env_communication_mode != "merge_minimum_uq":
            return pred_now

        n_agents = self.num_agents
        B = pred_now.shape[0]
        if B % n_agents != 0:
            return pred_now  # unexpected shape; skip rather than corrupt
        n_threads = B // n_agents
        D = pred_now.shape[-1]

        # Build the per-(thread, agent) receivers mask. The caller may pass per-thread scalars (whole
        # thread received) or per-agent arrays; both are accepted.
        receivers_mask = torch.zeros((n_threads, n_agents), dtype=torch.bool)
        for t_idx, flag in enumerate(list(received_comms_per_thread)[:n_threads]):
            if isinstance(flag, (list, tuple, np.ndarray, torch.Tensor)):
                receivers_mask[t_idx] = torch.as_tensor(np.asarray(flag), dtype=torch.bool).reshape(-1)[:n_agents]
            else:
                receivers_mask[t_idx] = bool(flag)

        if not receivers_mask.any():
            return pred_now

        pred_per = pred_now.reshape(n_threads, n_agents, D)
        uq_per = uncertainty_now.reshape(n_threads, n_agents, D)

        # Per-element argmin across agents within each thread → (n_threads, 1, D), then gather the
        # corresponding per-element predictions.
        best_idx = torch.argmin(uq_per, dim=1, keepdim=True)
        best_preds = torch.gather(pred_per, dim=1, index=best_idx)  # (n_threads, 1, D)

        # Merge only at receiver rows; non-receivers keep their own predictions.
        merge_select = receivers_mask.unsqueeze(-1)  # (n_threads, n_agents, 1)
        pred_per = torch.where(merge_select, best_preds.expand_as(pred_per), pred_per)
        pred_now = pred_per.reshape(B, D)

        # Persist the merged values at receiver rows of the autoregression source.
        receivers_flat = receivers_mask.reshape(-1)
        if prediction_prev is not None:
            prediction_prev[:, receivers_flat, -1, :] = pred_now[receivers_flat]

        return pred_now

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

        env = self.envs
        episodes = self.all_args.eval_episodes
        rollout_threads = self.n_eval_rollout_threads

        self.trainer.prep_rollout()

        # Call the reset function to determine whether this environment returns global observations.
        obs, share_obs, available_actions, infos = env.reset(return_info=True)
                
        # Get shape of observation space used by the predictor.
        use_observation_global = (
            "observation_global" in infos[0]
            and not self.all_args.disable_observation_global
        )
        if use_observation_global:
            obs_shape = get_shape_from_obs_space(self.buffer.share_obs_space)
        else:
            obs_shape = get_shape_from_obs_space(self.buffer.obs_space)

        # Set up the trajectory buffer for prediction.
        trajectory = TrajectoryBuffer(
            history_length=self.all_args.prediction_history_window,
            transition_includes_actions=self.all_args.prediction_history_include_actions,
        )

        prediction_prev = None  # For autoregression
        for i_episode in range(episodes):
            # Reset the environment and get the initial observations.
            obs, share_obs, available_actions, reset_infos = env.reset(return_info=True)
            rnn_states = torch.zeros((rollout_threads * self.num_agents, self.recurrent_N, self.hidden_size), dtype=torch.float32)
            masks = torch.ones((rollout_threads * self.num_agents, 1), dtype=torch.float32)
            self._seed_trajectory_buffer(trajectory, obs, rollout_threads, share_obs=share_obs)

            # Fallback (masked) observation_global used when prediction is not yet available.
            if use_observation_global and "observation_global" in reset_infos[0]:
                obs_global_fallback = np.array([info["observation_global"] for info in reset_infos])
                if self.all_args.observation_mask and "visibility_mask_global" in reset_infos[0]:
                    viz_mask_global_reset = np.array([info["visibility_mask_global"] for info in reset_infos])
                    obs_global_fallback = obs_global_fallback * viz_mask_global_reset
            else:
                obs_global_fallback = None

            # Unmasked ground truth from the previous step, used to compute true prediction error.
            ground_truth_obs_prev = None
            # Previous-iter infos, used to source received_comms flags for the ex-env comms merge.
            # None on the first iter of each episode → merge no-ops, matching the start-of-episode state.
            prev_infos = None

            dones = False
            j = -1
            while not np.all(dones):
                j += 1

                aa = np.concatenate(available_actions)
                if np.any(aa == None):
                    aa = None

                prediction, uncertainty, prediction_prev, prediction_error, use_prediction = self._compute_predictions(
                    trajectory,
                    prediction_prev,
                    rollout_threads,
                    obs_shape,
                    ground_truth_obs=ground_truth_obs_prev,
                )

                if use_prediction:
                    prediction_now = prediction[:, -1]
                    uncertainty_now = uncertainty[:, -1]
                    # Apply ex-env comms merge before stripping the action prefix so the writeback into
                    # prediction_prev (which has the full transition_dim) is shape-consistent.
                    received_comms = (
                        [info.get("received_comms", False) for info in prev_infos]
                        if prev_infos is not None else []
                    )
                    prediction_now = self._apply_ex_env_comms_merge(
                        prediction_now, uncertainty_now, received_comms, prediction_prev,
                    )
                    if self.all_args.prediction_history_include_actions:
                        act_size = int(np.prod(get_shape_from_act_space(self.buffer.act_space)))
                        prediction_now = prediction_now[..., act_size:]
                        uncertainty_now = uncertainty_now[..., act_size:]
                else:
                    prediction_now = None
                    uncertainty_now = torch.zeros((rollout_threads, self.num_agents, *obs_shape), dtype=torch.float32)

                # Reshape obs from [n_threads, n_agents, obs_dim] to [n_threads*n_agents, obs_dim]
                # to match the format expected by the policy (same as buffer.compatibility_get_policy_input).
                if obs.dtype == object:
                    obs_policy = obs.reshape(-1, *obs.shape[-1:])
                else:
                    obs_policy = torch.from_numpy(obs.reshape(-1, *obs.shape[2:])).float()

                replace_with_pred = use_prediction and self.all_args.prediction_during_training
                append_uq = self.all_args.prediction_uq_injection_method == "append"
                if self.all_args.state_encoder and replace_with_pred:
                    global_obs = prediction_now
                    if append_uq:
                        global_obs = torch.cat([global_obs, uncertainty_now], dim=-1)
                else:
                    if self.all_args.state_encoder and obs_global_fallback is not None:
                        global_obs = torch.from_numpy(
                            obs_global_fallback.reshape(rollout_threads * self.num_agents, *obs_global_fallback.shape[2:])
                        ).float()
                        if append_uq:
                            uq_pad = uncertainty_now if use_prediction else torch.zeros_like(global_obs)
                            global_obs = torch.cat([global_obs, uq_pad], dim=-1)
                    else:
                        global_obs = None
                    # For GNN envs (object-dtype obs), the prediction cannot replace the graph obs directly.
                    # The prediction (from observation_global) can only feed the state encoder.
                    if replace_with_pred and obs.dtype != object:
                        obs_policy = prediction_now
                        if append_uq:
                            obs_policy = torch.cat([obs_policy, uncertainty_now], dim=-1)
                    elif obs.dtype != object and append_uq:
                        uq_pad = uncertainty_now if use_prediction else torch.zeros_like(obs_policy)
                        obs_policy = torch.cat([obs_policy, uq_pad], dim=-1)


                actions, rnn_states = self.trainer.policy.act(
                    obs_policy,
                    rnn_states,
                    masks,
                    deterministic=False,
                    available_actions=aa,
                    global_obs=global_obs
                )

                # Prepare the actions for the environment.
                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                actions = actions.detach().cpu().reshape((rollout_threads, self.num_agents, *actions.shape[1:]))
                rnn_states = rnn_states.detach()
                actions_env = [actions[idx, :, :].numpy() for idx in range(rollout_threads)]

                obs, share_obs, rewards, dones, infos, available_actions = env.step(actions_env)

                observation_global = (
                    np.array([info["observation_global"] for info in infos])
                    if use_observation_global and "observation_global" in infos[0]
                    else None
                )
                # Save unmasked ground truth before masking, for use as prediction error reference next step.
                if observation_global is not None:
                    ground_truth_obs_prev = observation_global.copy()
                elif obs.dtype != object:
                    ground_truth_obs_prev = obs.copy()
                else:
                    ground_truth_obs_prev = None
                # Cache infos so the next iter's merge can read received_comms from this env transition.
                prev_infos = infos
                if hasattr(self.policy, "predictors") and not self.all_args.prediction_disable:
                    obs, observation_global = self._apply_masks_and_update_trajectory(
                        trajectory,
                        obs,
                        infos,
                        actions,
                        rollout_threads,
                        global_obs=observation_global,
                    )
                else:
                    if self.all_args.observation_mask:
                        viz_mask_local = np.array([info["visibility_mask"] for info in infos]) if "visibility_mask" in infos[0] else np.ones_like(obs)
                        viz_mask_global = np.array([info["visibility_mask_global"] for info in infos]) if "visibility_mask_global" in infos[0] else None
                        if obs.dtype != object:
                            obs = obs * viz_mask_local
                        if use_observation_global and observation_global is not None and viz_mask_global is not None:
                            observation_global = observation_global * viz_mask_global

                # Save the current masked observation_global for use as fallback in the next step.
                obs_global_fallback = observation_global

                # Add prediction error to infos for logging.
                infos[0]["prediction_error_mean"] = prediction_error.mean().item() if prediction_error is not None else 0.0
                infos[0]["prediction_trajectory_error_mean"] = prediction_error.mean().item() if prediction_error is not None else 0.0

                # Add uncertainty to infos for logging.
                infos[0]["prediction_uncertainty_mean"] = uncertainty_now.mean().item()
                if prediction_error is not None:
                    uncertainty_gap = uncertainty_now - prediction_error
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
                                episodes,
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

        env = self.envs
        episodes = self.all_args.render_episodes
        rollout_threads = self.n_render_rollout_threads

        self.trainer.prep_rollout()

        # Call the reset function to determine whether this environment returns global observations.
        obs, share_obs, available_actions, infos = env.reset(return_info=True)
                
        # Get shape of observation space used by the predictor.
        use_observation_global = (
            "observation_global" in infos[0]
            and not self.all_args.disable_observation_global
        )
        if use_observation_global:
            obs_shape = get_shape_from_obs_space(self.buffer.share_obs_space)
        else:
            obs_shape = get_shape_from_obs_space(self.buffer.obs_space)

        # Set up the trajectory buffer for prediction.
        trajectory = TrajectoryBuffer(
            history_length=self.all_args.prediction_history_window,
            transition_includes_actions=self.all_args.prediction_history_include_actions,
        )

        prediction_prev = None  # For autoregression
        for i_episode in range(episodes):
            # Reset the environment and get the initial observations.
            obs, share_obs, available_actions, reset_infos = env.reset(return_info=True)
            rnn_states = torch.zeros((rollout_threads * self.num_agents, self.recurrent_N, self.hidden_size), dtype=torch.float32)
            masks = torch.ones((rollout_threads * self.num_agents, 1), dtype=torch.float32)
            self._seed_trajectory_buffer(trajectory, obs, rollout_threads, share_obs=share_obs)

            # Fallback (masked) observation_global used when prediction is not yet available.
            if use_observation_global and "observation_global" in reset_infos[0]:
                obs_global_fallback = np.array([info["observation_global"] for info in reset_infos])
                if self.all_args.observation_mask and "visibility_mask_global" in reset_infos[0]:
                    viz_mask_global_reset = np.array([info["visibility_mask_global"] for info in reset_infos])
                    obs_global_fallback = obs_global_fallback * viz_mask_global_reset
            else:
                obs_global_fallback = None

            if self.all_args.save_gifs:        
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            # Unmasked ground truth from the previous step, used to compute true prediction error.
            ground_truth_obs_prev = None
            # Previous-iter infos, used to source received_comms flags for the ex-env comms merge.
            # None on the first iter of each episode → merge no-ops, matching the start-of-episode state.
            prev_infos = None

            dones = False
            reward_total = 0.0
            while not np.all(dones):
                time_start = time.time()

                aa = np.concatenate(available_actions)
                if np.any(aa == None):
                    aa = None

                prediction, uncertainty, prediction_prev, prediction_error, use_prediction = self._compute_predictions(
                    trajectory,
                    prediction_prev,
                    rollout_threads,
                    obs_shape,
                    ground_truth_obs=ground_truth_obs_prev,
                )

                if use_prediction:
                    prediction_now = prediction[:, -1]
                    uncertainty_now = uncertainty[:, -1]
                    # Apply ex-env comms merge before stripping the action prefix so the writeback into
                    # prediction_prev (which has the full transition_dim) is shape-consistent.
                    received_comms = (
                        [info.get("received_comms", False) for info in prev_infos]
                        if prev_infos is not None else []
                    )
                    prediction_now = self._apply_ex_env_comms_merge(
                        prediction_now, uncertainty_now, received_comms, prediction_prev,
                    )
                    if self.all_args.prediction_history_include_actions:
                        act_size = int(np.prod(get_shape_from_act_space(self.buffer.act_space)))
                        prediction_now = prediction_now[..., act_size:]
                        uncertainty_now = uncertainty_now[..., act_size:]

                    for agentIdx in range(self.num_agents):
                        print(f"Mean Prediction Error ({agentIdx}): {prediction_error.mean():.2f}")
                        # print(f"Mean Trajectory Prediction Error ({agentIdx}): {prediction_error.mean():.2f}")
                else:
                    prediction_now = None
                    uncertainty_now = torch.zeros((rollout_threads, self.num_agents, *obs_shape), dtype=torch.float32)

                # Reshape obs from [n_threads, n_agents, obs_dim] to [n_threads*n_agents, obs_dim]
                # to match the format expected by the policy (same as buffer.compatibility_get_policy_input).
                if obs.dtype == object:
                    obs_policy = obs.reshape(-1, *obs.shape[-1:])
                else:
                    obs_policy = torch.from_numpy(obs.reshape(-1, *obs.shape[2:])).float()

                replace_with_pred = use_prediction and self.all_args.prediction_during_training
                append_uq = self.all_args.prediction_uq_injection_method == "append"
                if self.all_args.state_encoder and replace_with_pred:
                    global_obs = prediction_now
                    if append_uq:
                        global_obs = torch.cat([global_obs, uncertainty_now], dim=-1)
                else:
                    if self.all_args.state_encoder and obs_global_fallback is not None:
                        global_obs = torch.from_numpy(
                            obs_global_fallback.reshape(rollout_threads * self.num_agents, *obs_global_fallback.shape[2:])
                        ).float()
                        if append_uq:
                            uq_pad = uncertainty_now if use_prediction else torch.zeros_like(global_obs)
                            global_obs = torch.cat([global_obs, uq_pad], dim=-1)
                    else:
                        global_obs = None
                    # For GNN envs (object-dtype obs), the prediction cannot replace the graph obs directly.
                    # The prediction (from observation_global) can only feed the state encoder.
                    if replace_with_pred and obs.dtype != object:
                        obs_policy = prediction_now
                        if append_uq:
                            obs_policy = torch.cat([obs_policy, uncertainty_now], dim=-1)
                    elif obs.dtype != object and append_uq:
                        uq_pad = uncertainty_now if use_prediction else torch.zeros_like(obs_policy)
                        obs_policy = torch.cat([obs_policy, uq_pad], dim=-1)


                actions, rnn_states = self.trainer.policy.act(
                    obs_policy,
                    rnn_states,
                    masks,
                    deterministic=False,
                    available_actions=aa,
                    global_obs=global_obs
                )

                # Prepare the actions for the environment.
                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                actions = actions.detach().cpu().reshape((rollout_threads, self.num_agents, *actions.shape[1:]))
                rnn_states = rnn_states.detach()
                actions_env = [actions[idx, :, :].numpy() for idx in range(rollout_threads)]

                obs, share_obs, rewards, dones, infos, available_actions = env.step(actions_env)

                observation_global = (
                    np.array([info["observation_global"] for info in infos])
                    if use_observation_global and "observation_global" in infos[0]
                    else None
                )
                # Save unmasked ground truth before masking, for use as prediction error reference next step.
                if observation_global is not None:
                    ground_truth_obs_prev = observation_global.copy()
                elif obs.dtype != object:
                    ground_truth_obs_prev = obs.copy()
                else:
                    ground_truth_obs_prev = None
                # Cache infos so the next iter's merge can read received_comms from this env transition.
                prev_infos = infos
                if hasattr(self.policy, "predictors") and not self.all_args.prediction_disable:
                    obs, observation_global = self._apply_masks_and_update_trajectory(
                        trajectory,
                        obs,
                        infos,
                        actions,
                        rollout_threads,
                        global_obs=observation_global,
                    )
                else:
                    if self.all_args.observation_mask:
                        viz_mask_local = np.array([info["visibility_mask"] for info in infos]) if "visibility_mask" in infos[0] else np.ones_like(obs)
                        viz_mask_global = np.array([info["visibility_mask_global"] for info in infos]) if "visibility_mask_global" in infos[0] else None
                        if obs.dtype != object:
                            obs = obs * viz_mask_local
                        if use_observation_global and observation_global is not None and viz_mask_global is not None:
                            observation_global = observation_global * viz_mask_global

                # Save the current masked observation_global for use as fallback in the next step.
                obs_global_fallback = observation_global

                reward_total += rewards[0].sum()

                time_stop = time.time()

                # Perform rendering.
                if ipython_clear_output:
                    clear_output(wait = True)
                spf = prediction_now if use_prediction else None
                env.envs[0].env.render(spf, history_length=self.all_args.prediction_history_window, uncertainty=uncertainty_now)

                # append frame
                if self.all_args.save_gifs:        
                    image = infos[0]["frame"]
                    frames.append(image)
                
                # Print the FPS information.
                print(f"Step {env.envs[0].env.step_count} - FPS: {1 / (time_stop - time_start):.2f} (excluding render) - Reward Total: {reward_total:.2f}")

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )
