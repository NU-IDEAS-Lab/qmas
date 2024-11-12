from gymnasium import spaces

from .model_base import CommunicationBaseModel

class BooleanObservationModel(CommunicationBaseModel):
    ''' This model uses predefined messages - the sender's observations.
        The only decision is a boolean whether to communicate or not. '''

    def communication_message_space(self, env, agent):
        ''' Returns the communication space. '''

        return env.observation_space(agent)
    
    
    def communication_action_space(self, env, agent):
        ''' Returns the communication action space. '''

        return spaces.Discrete(2)

    
    def on_communicate_action(self, env, agent, communication):
        ''' Processes the communication action. '''

        # Check whether we should communicate.
        if communication == 1:
            message = env.observe(agent)

            # Perform broadcast communication.
            for neighbor in env.possible_agents:
                if agent != neighbor:
                    self.communicate(agent, neighbor, message)