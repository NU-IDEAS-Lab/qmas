from gymnasium import spaces
import functools
import numpy as np

from .network import CommunicationNetwork

class CommunicationBaseModel:
    ''' Defines the base model for the communication module. '''

    def __init__(self, agent_position_fn=None):
        ''' Initializes the communication module. '''
        self.network = CommunicationNetwork()

        if agent_position_fn is None:
            agent_position_fn = lambda env, agent: agent.position
        self.agent_position_fn = agent_position_fn
    

    @functools.lru_cache(maxsize=None)
    def communication_message_space(self, env, agent):
        ''' Returns the communication space. '''

        raise NotImplementedError('This method must be implemented in the derived class.')


    @functools.lru_cache(maxsize=None)
    def communication_action_space(self, env, agent):
        ''' Returns the communication action space. '''

        raise NotImplementedError('This method must be implemented in the derived class.')


    @functools.lru_cache(maxsize=None)
    def available_communication_actions_space(self, env, agent):
        ''' Returns the available communication action space. '''

        action_space = self.communication_action_space(env, agent)
        def get_available_action_space(action_space):
            if action_space.__class__.__name__ in ["Tuple", "Dict"]:
                return spaces.Dict({k: get_available_action_space(v) for k, v in action_space.spaces.items()})
            elif action_space.__class__.__name__ == "Discrete":
                return spaces.MultiBinary(action_space.n)
            elif action_space.__class__.__name__ == "Box" and np.issubdtype(action_space.dtype, np.integer):
                return spaces.MultiBinary(action_space.high - action_space.low + 1)
            elif action_space.__class__.__name__ == "MultiDiscrete":
                return spaces.MultiBinary(len(action_space.nvec), max(action_space.nvec))
            else:
                raise NotImplementedError(f"Action space {action_space} not supported for action masking.")
        res = get_available_action_space(action_space)
        res = spaces.flatten_space(res)
        return res


    def available_communication_actions(self, env, agent):
        ''' Returns the available communication action. '''

        space = self.available_communication_actions_space(env, agent)
        return np.ones(space.shape, dtype=space.dtype)


    def __call__(self, env, agent):
        ''' Returns the communications received by the agent. '''
        
        return self.communicated_messages_get(agent)


    def on_reset(self, env):
        ''' Resets the communication network upon environment reset. '''

        self.network.clear()

        # Add nodes.
        for agent in env.possible_agents:
            self.network.add_node(agent, position=self.agent_position_fn(env, agent))
        
        # Add edges.
        for agent in env.possible_agents:
            for neighbor in env.possible_agents:
                if agent != neighbor:
                    self.network.add_edge(agent, neighbor, qos=self.network.calculate_edge_qos(agent, neighbor))
        
        # Reset communicated messages.
        self.communicated_messages_reset(env, env.possible_agents)
    
    
    def on_step(self, env):
        ''' Updates the communication network upon environment step. '''

        # Reset communicated messages.
        self.communicated_messages_reset(env, env.possible_agents)

        # Update positions.
        for agent in env.agents:
            self.network.nodes[agent]["position"] = self.agent_position_fn(env, agent)

        # Update QoS.
        for agent in env.agents:
            for neighbor in env.agents:
                if agent != neighbor:
                    self.network.edges[agent, neighbor]["qos"] = self.network.calculate_edge_qos(agent, neighbor)


    def on_communicate_action(env, agent, communication):
        ''' Processes the communication action. '''

        raise NotImplementedError('This method must be implemented in the derived class.')


    def communicate(self, sender, receiver, message):
        ''' Communicates a message from the sender to the receiver. '''

        self.communications[receiver][sender] = message
        self.messages_sent += 1


    def communicated_messages_get(self, agent):
        ''' Returns messages which were sent to the agent in this time step. '''

        return self.communications[agent]


    def communicated_messages_get_count(self):
        ''' Returns the number of messages sent in this time step. '''

        return self.messages_sent


    def communicated_messages_reset(self, env, possible_agents):
        ''' Resets the communicated messages. '''

        self.communications = {}
        self.messages_sent = 0
        for agent in possible_agents:
            self.communications[agent] = {}
            for neighbor in possible_agents:
                if neighbor != agent:
                    message_space = self.communication_message_space(env, neighbor)
                    self.communications[agent][neighbor] = np.zeros(message_space.shape, dtype=message_space.dtype)
