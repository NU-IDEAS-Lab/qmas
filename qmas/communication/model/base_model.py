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