import functools
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper

class CommunicationWrapper(BaseParallelWrapper):
    ''' Performs communication between agents and adds to agents' observations. '''

    def __init__(self, env, communication_model):
        super().__init__(env)
        self.communication_model = communication_model

        # Literally replace the environment's observation function with our own.
        self.env._observe_original = self.env.observe
        self.env.observe = self.observe
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        ''' Adds communication space to the action space. '''

        action_space = super().action_space(agent)
        comms_action_space = self.communication_model.communication_action_space(self.env, agent)

        if isinstance(action_space, spaces.Dict):
            action_space.spaces["communication"] = comms_action_space
            res = action_space
        else:
            action_space_dict = spaces.Dict({
                "action": action_space,
                "communication": comms_action_space
            })
            res = action_space_dict
        
        # Convert any Discrete spaces to Box spaces.
        self.converted_spaces = {}
        for k, v in res.spaces.items():
            self.converted_spaces[k] = False
            if isinstance(v, spaces.Discrete):
                res.spaces[k] = spaces.Box(low=v.start, high=v.start + v.n - 1, dtype=int)
                self.converted_spaces[k] = True
        
        return res


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        ''' Adds communication space to the observation space. '''

        obs_space = super().observation_space(agent)

        # The received communication space is the communication space of all other agents.
        received_communication_space = spaces.Dict({
            a: self.communication_model.communication_message_space(self.env, a) for a in self.agents if a != agent
        })
        received_communication_space = spaces.flatten_space(received_communication_space)

        if isinstance(obs_space, spaces.Dict):
            obs_space.spaces["communication"] = received_communication_space
            # Ensure ordering.
            obs_space.spaces = dict(sorted(obs_space.spaces.items()))
            return obs_space
        else:
            obs_space_dict = spaces.Dict({
                "communication": received_communication_space,
                "observation": obs_space
            })
            return obs_space_dict
    

    def reset(self, *args, **kwargs):
        ''' Resets the environment and communication network. '''

        self.communication_model.communicated_messages_reset(self.env, self.env.possible_agents)
        res = super().reset(*args, **kwargs)
        self.communication_model.on_reset(self.env)
        return res


    def observe(self, agent):
        ''' Adds communicated information to the observation. '''

        received_communication_space = spaces.Dict({
            a: self.communication_model.communication_message_space(self.env, a) for a in self.agents if a != agent
        })
        communications = spaces.flatten(received_communication_space, self.communication_model(self.env, agent))

        obs = self.env._observe_original(agent)
        if isinstance(obs, dict):
            obs["communication"] = communications

            # Ensure dictionary ordering.
            obs = dict(sorted(obs.items()))
            return obs
        else:
            return {
                "communication": communications,
                "observation": obs
            }
    

    @functools.lru_cache(maxsize=None)
    def available_actions_space(self, agent):
        ''' Adds communication space to the available actions space. '''

        available_space = self.env.available_actions_space(agent)

        if isinstance(available_space, spaces.Dict):
            available_space.spaces["communication"] = self.communication_model.available_communication_actions_space(self.env, agent)
            return available_space
        else:
            available_space_dict = spaces.Dict({
                "action": available_space,
                "communication": self.communication_model.available_communication_actions_space(self.env, agent)
            })
            return available_space_dict


    def available_actions(self, agent):
        ''' Adds communication action to the available actions. '''

        # if not hasattr(self.env, "available_actions") or not callable(self.env.available_actions):
        #     return None
        available = self.env.available_actions(agent)

        if isinstance(available, dict):
            available["communication"] = self.communication_model.available_communication_actions(self.env, agent)
            res = available
        else:
            res = {
                "action": available,
                "communication": self.communication_model.available_communication_actions(self.env, agent)
            }
        res = spaces.flatten(self.available_actions_space(agent), res)
        return res


    def step(self, action, *args, **kwargs):
        ''' Updates the communication network upon environment step. '''

        # Process the action.
        action_new = {}
        for agent in self.agents:
            if agent in action:
                # Unflatten the action.
                action_space = self.action_space(agent)
                act = spaces.unflatten(action_space, action[agent])

                # Process the communication.
                communication = act.pop("communication")
                self.communication_model.on_communicate_action(self.env, agent, communication)

                # Restore the action to what the underlying environment expects.
                if "action" in action_space.spaces:
                    action_space_new = action_space["action"]
                    act_new = act["action"]
                else:
                    # This happens when there was an existing dictionary action space that we added "communication" to.
                    raise NotImplementedError("Dict action spaces not currently supported.")
                action_new[agent] = spaces.flatten(action_space_new, act_new)

                # De-convert any Discrete spaces that we had converted to box.
                # TODO: This is super messy.
                if self.converted_spaces["action"]:
                    action_new[agent] = action_new[agent][0]
        
        # Count messages sent.
        messages_sent = self.communication_model.communicated_messages_get_count()

        # Perform environment step.
        obs, reward, terminated, truncated, info = self.env.step(action_new, *args, **kwargs)

        # Update the communication model.
        self.communication_model.on_step(self.env)

        # Add information.
        info["messages_sent"] = messages_sent

        return obs, reward, terminated, truncated, info