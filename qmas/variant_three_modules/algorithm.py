from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO

import numpy as np

class QmasAlgorithm(R_MAPPO):
    """
    Trainer class for QMAS to update policies.
    """

    def train(self, buffer, update_actor=True, update_critic=True, last_step=-1):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.
        :param update_critic: (bool) whether to update critic network.
        :param last_step: (int) last step of the episode. Defaults to -1, which will use all steps in the buffer.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:last_step] - self.value_normalizer.denormalize(buffer.value_preds[:last_step])
        else:
            advantages = buffer.returns[:last_step] - buffer.value_preds[:last_step]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:last_step] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        num_updates = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length, last_step=last_step)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch, last_step=last_step)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch, last_step=last_step)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor, update_critic)
                
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

                comms_loss, comms_grad_norm = self.comms_update(sample)
                train_info['comms_loss'] = comms_loss.item()
                train_info['comms_grad_norm'] = comms_grad_norm

                num_updates += 1

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.comms.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.comms.eval()
