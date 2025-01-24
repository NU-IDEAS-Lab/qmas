from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO

import numpy as np

class QmasAlgorithm(R_MAPPO):
    """
    Trainer class for QMAS to update policies.
    """

    def train_initialize_info(self, train_info):
        super().train_initialize_info(train_info)
        train_info['comms_loss'] = 0
        train_info['comms_grad_norm'] = 0


    def train_sample(self, sample, train_info, update_actor=True, update_critic=True):
        ''' Performs update for a single sample. '''
        
        super().train_sample(sample, train_info, update_actor, update_critic)

        comms_loss, comms_grad_norm = self.comms_update(sample)
        train_info['comms_loss'] = comms_loss.item()
        train_info['comms_grad_norm'] = comms_grad_norm


    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.comms.train()


    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.comms.eval()
