import torch
import numpy as np

from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack
from torchrl.data.replay_buffers import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, RandomSampler

from onpolicy.utils.shared_buffer import SharedReplayBuffer as SharedReplayBufferOld


class FixedSamplerWithoutReplacement(SamplerWithoutReplacement):
    """This class fixes the SamplerWithoutReplacement class to allow both drop_last and shuffle at the same time.
    
    A data-consuming sampler that ensures that the same sample is not present in consecutive batches.

    Args:
        drop_last (bool, optional): if ``True``, the last incomplete sample (if any) will be dropped.
            If ``False``, this last sample will be kept and (unlike with torch dataloaders)
            completed with other samples from a fresh indices permutation.
            Defaults to ``False``.
        shuffle (bool, optional): if ``False``, the items are not randomly
            permuted. This enables to iterate over the replay buffer in the
            order the data was collected. Defaults to ``True``.

    *Caution*: If the size of the storage changes in between two calls, the samples will be re-shuffled
    (as we can't generally keep track of which samples have been sampled before and which haven't).

    Similarly, it is expected that the storage content remains the same in between two calls,
    but this is not enforced.

    When the sampler reaches the end of the list of available indices, a new sample order
    will be generated and the resulting indices will be completed with this new draw, which
    can lead to duplicated indices, unless the :obj:`drop_last` argument is set to ``True``.

    """

    def __init__(self, drop_last: bool = False, shuffle: bool = True):
        # Set drop_last to false, since we have made a simpler version of that mechanism.
        # super().__init__(drop_last=False, shuffle=shuffle)
        super().__init__(drop_last=drop_last, shuffle=shuffle)

    def _storage_len(self, storage):
        length = len(storage)
        if self.drop_last and length > 0:
            return length - 1
        return length

# class TrajectorySampler(RandomSampler):
#     ''' Samples trajectories of contiguous data from the replay buffer. '''

#     def __init__(self, *args, trajectory_length=1, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.trajectory_length = trajectory_length
    
#     def sample(self, storage, batch_size):
#         ''' Samples a batch of trajectories. '''

#         ''' Samples a trajectory of data from the replay buffer. '''
#         if len(storage) == 0:
#             return None

#         # Sample a random index.
#         idx_start = torch.randint(0, len(storage) - self.trajectory_length + 1, (batch_size,))
#         idx_end = idx_start + self.trajectory_length
#         return idx, {}


class SharedReplayBuffer(TensorDictReplayBuffer, SharedReplayBufferOld):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_gae_amadm = args.use_gae_amadm
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.obs_object = False
        self.share_obs_object = False

        # Set up the buffer.
        super(TensorDictReplayBuffer, self).__init__(
            batch_size=args.episode_length // args.num_mini_batch,
            storage=LazyTensorStorage(
                max_size=args.episode_length + 1,
                ndim=1
            ),
            # sampler=FixedSamplerWithoutReplacement(
            sampler=SamplerWithoutReplacement(
                shuffle=False,
                drop_last=True
            )
            # sampler=TrajectorySampler(
            #     trajectory_length=5
            # )
        )

        print(f"Buffer initialized with episode length {args.episode_length}")


    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, delta_steps=None, available_actions=None,
               visibility_mask=None, legacy_mode=True):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        :param delta_steps: (np.ndarray) number of steps since last update.
        :param visibility_mask: (np.ndarray) visibility mask for agent observations, if applicable.
        :param legacy_mode: (bool) whether to use legacy mode for inserting data. Will use timesteps t and t+1.
        """

        if bad_masks is None:
            bad_masks = np.ones_like(masks)
        if active_masks is None:
            active_masks = np.ones_like(masks)
        if np.any(available_actions == None):
            available_actions = np.ones_like(actions)
        if delta_steps is None:
            delta_steps = np.ones_like(value_preds)
        if visibility_mask is None:
            visibility_mask = np.ones_like(obs)
        
        # Convert any np.object arrays to tensors of NonTensorData.
        if isinstance(obs, np.ndarray) and obs.dtype == object:
            self.obs_object = True
            obs = NonTensorStack(NonTensorData(
                obs,
                batch_size=torch.Size([]),
                device='cpu',
                names=None,
            ))
        if isinstance(share_obs, np.ndarray) and share_obs.dtype == object:
            self.share_obs_object = True
            share_obs = NonTensorStack(NonTensorData(
                share_obs,
                batch_size=torch.Size([]),
                device='cpu',
                names=None,
            ))

        # Create a tensordict of all the data.
        data = TensorDict({
            'share_obs': share_obs, #+1
            'obs': obs, #+1
            'rnn_states_actor': rnn_states_actor, #+1
            'rnn_states_critic': rnn_states_critic, #+1
            'actions': actions,
            'action_log_probs': action_log_probs,
            'value_preds': value_preds,
            'rewards': rewards,
            'returns': np.zeros_like(rewards),
            'masks': masks, #+1
            'bad_masks': bad_masks, #+1
            'active_masks': active_masks, #+1
            'delta_steps': delta_steps,
            'available_actions': available_actions, #+1
            'visibility_mask': visibility_mask, #+1
        })

        # In legacy mode, some data is added for timestep t, others for timestep t+1.
        if legacy_mode:
            # For step t, add to the existing data.
            
            if len(self) <= 0:
                # Special case for the first insertion.
                self.add(data)
            else:
                self["actions"][-1] = data["actions"]
                self["action_log_probs"][-1] = data["action_log_probs"]
                self["value_preds"][-1] = data["value_preds"]
                self["rewards"][-1] = data["rewards"]
                self["delta_steps"][-1] = data["delta_steps"]

                # Insert the data for t+1 into the buffer.
                # The t+1 step (`data`) will temporarily contain data for the previous (t) step.
                self.add(data)
        else:
            # In non-legacy mode, add all data for timestep t.
            self.add(data)


    def after_update(self, last_step=-1):
        """ Reset/clear the buffer. Called after update to model. """

        self.empty()


    def compute_returns(self, next_value, value_normalizer=None, last_step=-1):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """

        # if last_step == -1:
        last_step = len(self) - 1

        # Check whether we should use the AMADM GAE modification from https://arxiv.org/abs/2308.06036
        # Unfortunately, I don't have time to implement for all of the other options (like use_proper_time_limits),
        # so I just ignore them!
        if self._use_gae_amadm:
            self["value_preds"][last_step] = next_value
            gae = 0
            for step in reversed(range(last_step)):
                if self._use_popart or self._use_valuenorm:
                    delta = self["rewards"][step] + np.power(self.gamma, self["delta_steps"][step]) * value_normalizer.denormalize(
                        self["value_preds"][step + 1]) * self.masks[step + 1] \
                            - value_normalizer.denormalize(self["value_preds"][step])
                    gae = delta + np.power(self.gamma * self.gae_lambda, self["delta_steps"][step]) * self.masks[step + 1] * gae
                    self["returns"][step] = gae + value_normalizer.denormalize(self["value_preds"][step])
                else:
                    delta = self["rewards"][step] + np.power(self.gamma, self["delta_steps"][step]) * self["value_preds"][step + 1] * self.masks[step + 1] - \
                            self["value_preds"][step]
                    gae = delta + np.power(self.gamma * self.gae_lambda, self["delta_steps"][step]) * self.masks[step + 1] * gae
                    self["returns"][step] = gae + self["value_preds"][step]
        
        elif self._use_proper_time_limits:
            if self._use_gae:
                self["value_preds"][last_step] = next_value
                gae = 0
                for step in reversed(range(self["rewards"].shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self["rewards"][step] + self.gamma * value_normalizer.denormalize(
                            self["value_preds"][step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self["value_preds"][step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self["returns"][step] = gae + value_normalizer.denormalize(self["value_preds"][step])
                    else:
                        delta = self["rewards"][step] + self.gamma * self["value_preds"][step + 1] * self.masks[step + 1] - \
                                self["value_preds"][step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self["returns"][step] = gae + self["value_preds"][step]
            else:
                self["returns"][last_step] = next_value
                for step in reversed(range(self["rewards"].shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self["returns"][step] = (self["returns"][step + 1] * self.gamma * self.masks[step + 1] + self["rewards"][
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self["value_preds"][step])
                    else:
                        self["returns"][step] = (self["returns"][step + 1] * self.gamma * self.masks[step + 1] + self["rewards"][
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self["value_preds"][step]
        else:
            if self._use_gae:
                self["value_preds"][last_step] = next_value
                gae = 0
                for step in reversed(range(self["rewards"].shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        if self.algo == "mat" or self.algo == "mat_dec":
                            value_t = value_normalizer.denormalize(self["value_preds"][step])
                            value_t_next = value_normalizer.denormalize(self["value_preds"][step + 1])
                            rewards_t = self["rewards"][step]

                            # mean_v_t = np.mean(value_t, axis=-2, keepdims=True)
                            # mean_v_t_next = np.mean(value_t_next, axis=-2, keepdims=True)
                            # delta = rewards_t + self.gamma * self.masks[step + 1] * mean_v_t_next - mean_v_t

                            delta = rewards_t + self.gamma * self.masks[step + 1] * value_t_next - value_t
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self.advantages[step] = gae
                            self["returns"][step] = gae + value_t
                        else:
                            delta = self["rewards"][step] + self.gamma * value_normalizer.denormalize(
                                self["value_preds"][step + 1]) * self.masks[step + 1] \
                                    - value_normalizer.denormalize(self["value_preds"][step])
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self["returns"][step] = gae + value_normalizer.denormalize(self["value_preds"][step])
                    else:
                        if self.algo == "mat" or self.algo == "mat_dec":
                            rewards_t = self["rewards"][step]
                            mean_v_t = np.mean(self["value_preds"][step], axis=-2, keepdims=True)
                            mean_v_t_next = np.mean(self["value_preds"][step + 1], axis=-2, keepdims=True)
                            delta = rewards_t + self.gamma * self.masks[step + 1] * mean_v_t_next - mean_v_t

                            # delta = rewards_t + self.gamma * self["value_preds"][step + 1] * \
                            #         self.masks[step + 1] - self["value_preds"][step]
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self.advantages[step] = gae
                            self["returns"][step] = gae + self["value_preds"][step]

                        else:
                            delta = self["rewards"][step] + self.gamma * self["value_preds"][step + 1] * \
                                    self.masks[step + 1] - self["value_preds"][step]
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self["returns"][step] = gae + self["value_preds"][step]
            else:
                self["returns"][last_step] = next_value
                for step in reversed(range(self["rewards"].shape[0])):
                    self["returns"][step] = self["returns"][step + 1] * self.gamma * self.masks[step + 1] + self["rewards"][step]


    def sample_trajectories(self, num_mini_batch, trajectory_size, legacy_mode=False):
        ''' Yields a batch of trajectories. '''

        storage_len = len(self.storage) - 1 # skip the last (incomplete) sample

        batch_size = (self.episode_length - trajectory_size) // num_mini_batch
        # idx_start = torch.randint(0, storage_len - trajectory_size, (batch_size,))
        idx_start = torch.randperm(storage_len - trajectory_size)
        idx_end = idx_start + trajectory_size

        # The storage is in the format of a single trajectory, so we need to split it into multiple trajectories.
        # Build the sample of shape (num_mini_batch, trajectory_size, ...)
        for i in range(0, len(idx_start), batch_size):
            res = []
            for j in range(i, min(i + batch_size, len(idx_start))):
                res.append(self.storage[idx_start[j]:idx_end[j]])
            res2 = torch.stack(res)

            if legacy_mode:
                yield self.compatibility_transform_sample(res2, index_shape=(batch_size, trajectory_size))
            else:
                yield res2


    ### COMPATIBILITY ###
    # Everything below here is for compatibility with the old replay buffer class.

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None, last_step=-1):
        sample, info = self.sample(return_info=True)
        sample["advantages"] = advantages[info["index"]]

        yield self.compatibility_transform_sample(sample)


    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length, last_step=-1):
        """
        We override this function to make it use torch.Tensor.permute instead of numpy.ndarray.transpose.

        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """

        def _flatten(T, N, x):
            return x.reshape(T * N, *x.shape[2:])


        def _cast(x):
            return x.permute(1, 2, 0, *range(3, x.ndim)).reshape(-1, *x.shape[3:])


        _, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        episode_length = len(self) - 1
        if last_step != -1:
            episode_length = last_step

        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:last_step].permute(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:last_step])
        if len(self.obs.shape) > 4:
            obs = self.obs[:last_step].permute(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            obs = _cast(self.obs[:last_step])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:last_step])
        returns = _cast(self.returns[:last_step])
        masks = _cast(self.masks[:last_step])
        active_masks = _cast(self.active_masks[:last_step])
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:last_step].permute(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:last_step].permute(1, 2, 0, 3, 4).reshape(-1,
                                                                                         *self.rnn_states_critic.shape[
                                                                                          3:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:last_step])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)           
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch


    def compatibility_transform_sample(self, sample, index_shape=(-1,), data_start_dim=3):
        ''' Returns sample in format expected by existing onpolicy code. '''

        if self.share_obs_object:
            sample_share_obs = np.array(sample["share_obs"])
            share_obs_batch = sample_share_obs.reshape(*index_shape, *sample_share_obs.shape[data_start_dim+1:])
        else:
            share_obs_batch = sample["share_obs"].reshape(*index_shape, *sample["share_obs"].shape[2:])
        if self.obs_object:
            sample_obs = np.array(sample["obs"])
            obs_batch = sample_obs.reshape(*index_shape, *sample_obs.shape[data_start_dim+1:])
        else:
            obs_batch = sample["obs"].reshape(*index_shape, *sample["obs"].shape[data_start_dim:])
        rnn_states_batch = sample["rnn_states_actor"].reshape(*index_shape, *sample["rnn_states_actor"].shape[data_start_dim:])
        rnn_states_critic_batch = sample["rnn_states_critic"].reshape(*index_shape, *sample["rnn_states_critic"].shape[data_start_dim:])
        actions_batch = sample["actions"].reshape(*index_shape, *sample["actions"].shape[data_start_dim:])
        value_preds_batch = sample["value_preds"].reshape(*index_shape, 1)
        return_batch = sample["returns"].reshape(*index_shape, 1)
        masks_batch = sample["masks"].reshape(*index_shape, 1)
        active_masks_batch = sample["active_masks"].reshape(*index_shape, 1)
        old_action_log_probs_batch = sample["action_log_probs"].reshape(*index_shape, sample["action_log_probs"].shape[-1])
        adv_targ = sample["advantages"].reshape(*index_shape, 1)
        if np.any(sample["available_actions"] == None):
            available_actions_batch = None
        else:
            available_actions_batch = sample["available_actions"].reshape(*index_shape, sample["available_actions"].shape[-1])

        return share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch


    def compatibility_get_policy_input(self, step):
        ''' Gets the necessary policy input for a particular step, in the format expected by existing onpolicy code. '''
        sample = self[step]
        
        if self.share_obs_object:
            sample_share_obs = np.array(sample["share_obs"])
            share_obs = sample_share_obs.reshape(-1, *sample_share_obs.shape[4:])
        else:
            share_obs = np.concatenate(sample["share_obs"].numpy())
        if self.obs_object:
            sample_obs = np.array(sample["obs"])
            obs = sample_obs.reshape(-1, *sample_obs.shape[4:], 1)
        else:
            obs = np.concatenate(sample["obs"].numpy())
        rnn_states_actor = np.concatenate(sample["rnn_states_actor"].numpy())
        rnn_states_critic = np.concatenate(sample["rnn_states_critic"].numpy())
        masks = np.concatenate(sample["masks"].numpy())
        if np.any(sample["available_actions"] == None):
            available_actions = None
        else:
            available_actions = np.concatenate(sample["available_actions"].numpy())

        return share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions


    @property
    def share_obs(self):
        return self["share_obs"]

    @property
    def obs(self):
        return self["obs"]
    
    @property
    def rnn_states(self):
        return self["rnn_states_actor"]

    @property
    def rnn_states_actor(self):
        return self["rnn_states_actor"]
    
    @property
    def rnn_states_critic(self):
        return self["rnn_states_critic"]
    
    @property
    def actions(self):
        return self["actions"]
    
    @property
    def action_log_probs(self):
        return self["action_log_probs"]
    
    @property
    def value_preds(self):
        return self["value_preds"]
    
    @property
    def rewards(self):
        return self["rewards"]
    
    @property
    def returns(self):
        return self["returns"]

    @property
    def masks(self):
        return self["masks"]
    
    @property
    def bad_masks(self):
        return self["bad_masks"]
    
    @property
    def active_masks(self):
        return self["active_masks"]
    
    @property
    def delta_steps(self):
        return self["delta_steps"]
    
    @property
    def available_actions(self):
        return self["available_actions"]