from gymnasium import spaces

from .network import CommunicationNetwork

class CommunicationBaseModel:
    ''' Defines the base model for the communication module. '''

    def __init__(self, args):
        ''' Initializes the communication module. '''
        self.args = args
        self.network = CommunicationNetwork()
    

    def communication_space(self):
        ''' Returns the communication space. '''

        raise NotImplementedError('This method must be implemented in the derived class.')


    def __call__(self, env, agent):
        ''' Returns the communication for the agent. '''
        
        return self.communicated_messages_get(agent)


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
        
        # Reset communicated messages.
        self.communicated_messages_reset()
    
    
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


    def on_communicate_action(env, agent, communication):
        ''' Processes the communication action. '''

        raise NotImplementedError('This method must be implemented in the derived class.')


    def communicate(self, sender, receiver, message):
        ''' Communicates a message from the sender to the receiver. '''

        self.communications[receiver][sender] = message


    def communicated_messages_get(self, agent):
        ''' Returns messages which were sent to the agent in this time step. '''

        return self.communications[agent]


    def communicated_messages_reset(self):
        ''' Resets the communicated messages. '''

        self.communications = {
            agent: {} for agent in self.env.possible_agents
        }