from gymnasium import spaces

from .model_base import CommunicationBaseModel

class BooleanObservationModel(CommunicationBaseModel):
    ''' This model uses predefined messages - the sender's observations.
        The only decision is a boolean whether to communicate or not. '''

    def communication_message_space(self, env, agent):
        ''' Returns the communication space. '''

        return spaces.Discrete(start=-1, n=len(env.pg.graph) + 1)
    
    
    def communication_action_space(self, env, agent):
        ''' Returns the communication action space. '''

        return spaces.Box(low=0, high=1, shape=(1, ), dtype=int)

    
    def on_communicate_action(self, env, agent, communication):
        ''' Processes the communication action. '''

        # Check whether we should communicate.
        if communication == 1:
            # We just tell people where we last visited.
            message = agent.lastNodeVisited
            if message is None:
                message = -1

            # Perform broadcast communication.
            for neighbor in env.possible_agents:
                if agent != neighbor:
                    self.communicate(agent, neighbor, message)