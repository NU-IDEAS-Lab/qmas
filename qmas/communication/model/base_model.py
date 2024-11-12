from gymnasium import spaces

from .network import CommunicationNetwork

class CommunicationBaseModel:
    ''' Defines the base model for the communication module. '''

    def __init__(self, args):
        ''' Initializes the communication module. '''
        self.args = args
        self.communication_space = spaces.Box(low=-1.0, high=1.0)
        self.network = CommunicationNetwork()


    def __call__(self, env, agent):
        ''' Returns the communication for the agent. '''
        raise NotImplementedError("Communication not implemented.")


    def on_reset(self, env):
        ''' Resets the communication network upon environment reset. '''

        self.network.clear()

        # Add nodes.
        for agent in env.possible_agents:
            self.network.add_node(agent, position=env.get_position(agent))
        
        # Add edges.
        for agent in env.possible_agents:
            for neighbor in env.possible_agents:
                if agent != neighbor:
                    self.network.add_edge(agent, neighbor, qos=self.network.calculate_edge_qos(agent, neighbor))
    
    
    def on_step(self, env):
        ''' Updates the communication network upon environment step. '''

        # Update positions.
        for agent in env.agents:
            self.network.get_node(agent)["position"] = env.get_position(agent)

        # Update QoS.
        for agent in env.agents:
            for neighbor in env.agents:
                if agent != neighbor:
                    self.network.get_edge(agent, neighbor)["qos"] = self.network.calculate_edge_qos(agent, neighbor)