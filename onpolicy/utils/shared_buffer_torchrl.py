import torch
import numpy as np

from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack
from torchrl.data.replay_buffers import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement


class SharedReplayBuffer(TensorDictReplayBuffer):
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

        self.obs_object = False
        self.share_obs_object = False

        # Set up the storage backend.
        storage = LazyTensorStorage(max_size=args.episode_length)

        # Set up the buffer.
        super().__init__(
            storage=storage,
            sampler=SamplerWithoutReplacement(
                shuffle=False
            )
        )

        print(f"Buffer initialized with episode length {args.episode_length}")


    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, delta_steps=None, available_actions=None):
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
        """

        if bad_masks is None:
            bad_masks = np.ones_like(masks)
        if active_masks is None:
            active_masks = np.ones_like(masks)
        if available_actions is None:
            available_actions = np.ones_like(actions)
        if delta_steps is None:
            delta_steps = np.ones_like(value_preds)
        
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

        # Create a tensordict of the data.
        data = TensorDict({
            'share_obs': share_obs,
            'obs': obs,
            'rnn_states_actor': rnn_states_actor,
            'rnn_states_critic': rnn_states_critic,
            'actions': actions,
            'action_log_probs': action_log_probs,
            'value_preds': value_preds,
            'rewards': rewards,
            'returns': np.zeros_like(rewards),
            'masks': masks,
            'bad_masks': bad_masks,
            'active_masks': active_masks,
            'delta_steps': delta_steps,
            'available_actions': available_actions
        })

        # Insert the data into the buffer.
        self.add(data)


    def after_update(self, last_step=-1):
        """Copy last timestep data to first index. Called after update to model."""

        # last_sample = self[-1]
        self.empty()
        # self.add(last_sample)


    def compute_returns(self, next_value, value_normalizer=None, last_step=-1):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """

        # if last_step == -1:
        #     last_step = self.episode_length
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


    ### COMPATIBILITY ###
    # Everything below here is for compatibility with the old replay buffer class.

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None, last_step=-1):
        batch_size = len(self) // num_mini_batch

        sample, info = self.sample(batch_size=batch_size, return_info=True)
        sample["advantages"] = advantages[info["index"]]

        yield self.compatibility_transform_sample(sample)


    def compatibility_transform_sample(self, sample):
        ''' Returns sample in format expected by existing onpolicy code. '''

        if self.share_obs_object:
            sample_share_obs = np.array(sample["share_obs"])
            share_obs_batch = sample_share_obs.reshape(-1, *sample_share_obs.shape[4:])
        else:
            share_obs_batch = sample["share_obs"].reshape(-1, *sample["share_obs"].shape[3:])
        if self.obs_object:
            sample_obs = np.array(sample["obs"])
            obs_batch = sample_obs.reshape(-1, *sample_obs.shape[4:])
        else:
            obs_batch = sample["obs"].reshape(-1, *sample["obs"].shape[3:])
        rnn_states_batch = sample["rnn_states_actor"].reshape(-1, *sample["rnn_states_actor"].shape[3:])
        rnn_states_critic_batch = sample["rnn_states_critic"]
        actions_batch = sample["actions"].reshape(-1, *sample["actions"].shape[3:])
        value_preds_batch = sample["value_preds"].reshape(-1)
        return_batch = sample["returns"].reshape(-1)
        masks_batch = sample["masks"].reshape(-1)
        active_masks_batch = sample["active_masks"].reshape(-1)
        old_action_log_probs_batch = sample["action_log_probs"].reshape(-1, sample["action_log_probs"].shape[-1])
        adv_targ = sample["advantages"].reshape(-1)
        available_actions_batch = sample["available_actions"].reshape(-1, sample["available_actions"].shape[-1])

        return share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch


    @property
    def share_obs(self):
        return self["share_obs"]

    @property
    def obs(self):
        return self["obs"]
    
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