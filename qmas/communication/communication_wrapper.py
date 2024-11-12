from gymnasium import spaces
from pettingzoo.utils.wrappers.base import BaseWrapper

class CommunicationWrapper(BaseWrapper):
    ''' Performs communication between agents and adds to agents' observations. '''

    def __init__(self, env, communication_model):
        super().__init__(env)
        self.communication_model = communication_model
        self.communication_space = communication_model.communication_space
    

    def action_space(self, agent):
        ''' Adds communication space to the action space. '''

        action_space = super().action_space(agent)
        if isinstance(action_space, spaces.Dict):
            action_space.spaces["communication"] = self.communication_space
            return action_space
        else:
            #TODO: May need a way to mark that this dictionary is not original to the environment.
            action_space_dict = spaces.Dict({
                "action": action_space,
                "communication": self.communication_space
            })
            return action_space_dict


    def observation_space(self, agent):
        ''' Adds communication space to the observation space. '''

        obs_space = super().observation_space(agent)
        received_communication_space = spaces.Dict({
            a: self.communication_space for a in self.agents if a != agent
        })
        if isinstance(obs_space, spaces.Dict):
            obs_space.spaces["communication"] = self.communication_space
            return obs_space
        else:
            obs_space_dict = spaces.Dict({
                "observation": obs_space,
                "communication": received_communication_space
            })
            return obs_space_dict
    

    def observe(self, agent):
        ''' Adds communicated information to the observation. '''

        obs = super().observe(agent)
        if isinstance(obs, dict):
            obs["communication"] = self.communication_model(self.env, agent)
            return obs
        else:
            return {
                "observation": obs,
                "communication": self.communication_model(self.env, agent)
            }