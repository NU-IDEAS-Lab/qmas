from gymnasium import spaces

from .network import CommunicationNetwork

class CommunicationBaseModel:
    ''' Defines the base model for the communication module. '''

    def __init__(self, agent_position_fn=None):
        ''' Initializes the communication module. '''
        self.network = CommunicationNetwork()

        if agent_position_fn is None:
            agent_position_fn = lambda env, agent: agent.position
        self.agent_position_fn = agent_position_fn
    

    def communication_message_space(self, env, agent):
        ''' Returns the communication space. '''

        raise NotImplementedError('This method must be implemented in the derived class.')


    def communication_action_space(self, env, agent):
        ''' Returns the communication action space. '''

        raise NotImplementedError('This method must be implemented in the derived class.')


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
        self.communicated_messages_reset(env.possible_agents)
    
    
    def on_step(self, env):
        ''' Updates the communication network upon environment step. '''

        # Reset communicated messages.
        self.communicated_messages_reset(env.possible_agents)

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


    def communicated_messages_get(self, agent):
        ''' Returns messages which were sent to the agent in this time step. '''

        return self.communications[agent]


    def communicated_messages_reset(self, possible_agents):
        ''' Resets the communicated messages. '''

        self.communications = {
            agent: {} for agent in possible_agents
        }