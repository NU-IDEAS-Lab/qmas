import functools
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper

class CommunicationWrapper(BaseParallelWrapper):
    ''' Performs communication between agents and adds to agents' observations. '''

    def __init__(self, env, communication_model):
        super().__init__(env)
        self.communication_model = communication_model
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        ''' Adds communication space to the action space. '''

        action_space = super().action_space(agent)
        comms_action_space = self.communication_model.communication_action_space(self.env, agent)

        if isinstance(action_space, spaces.Dict):
            action_space.spaces["communication"] = comms_action_space
            return action_space
        else:
            action_space_dict = spaces.Dict({
                "action": action_space,
                "communication": comms_action_space
            })
            return action_space_dict


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        ''' Adds communication space to the observation space. '''

        obs_space = super().observation_space(agent)

        # The received communication space is the communication space of all other agents.
        received_communication_space = spaces.Dict({
            a: self.communication_model.communication_message_space(self.env, a) for a in self.agents if a != agent
        })

        if isinstance(obs_space, spaces.Dict):
            obs_space.spaces["communication"] = received_communication_space
            return obs_space
        else:
            obs_space_dict = spaces.Dict({
                "observation": obs_space,
                "communication": received_communication_space
            })
            return obs_space_dict
    

    def reset(self, *args, **kwargs):
        ''' Resets the environment and communication network. '''

        res = super().reset(*args, **kwargs)
        self.communication_model.on_reset(self.env)
        return res


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
    

    def step(self, action):
        ''' Updates the communication network upon environment step. '''

        # Process the action.
        action_new = {}
        for agent in self.agents:
            if agent in action:
                # Process the communication.
                communication = action[agent].pop("communication")
                self.communication_model.on_communicate_action(self.env, agent, communication)

                # Restore the action to what the underlying environment expects.
                action_space = super().action_space(agent)
                if isinstance(action_space, spaces.Dict):
                    action_new[agent] = action[agent]
                else:
                    action_new[agent] = action[agent]["action"]

        # Perform environment step.
        ret = super().step(action_new)

        # Update the communication model.
        self.communication_model.on_step(self.env)
        return ret