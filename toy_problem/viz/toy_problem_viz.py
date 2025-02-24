# visualization for toy problem

import os
from gymnasium import spaces
import random
import numpy as np
import math
from copy import deepcopy
from matplotlib import pyplot as plt
import networkx as nx
from copy import copy
from enum import IntEnum
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data

def reformat():
    '''
    potential reformating of the data
    '''
    pass

class ToyVisualization():
    def __init__(self, env):
        self.obs_dict = env.obs_dict
        self.step_count = env.step_count
        self.reference = env.reference
        self.agents = env.agents
        self.reference_state_history = env.reference_state_history
    
    def _ready(self, interval = 20):
        '''
        set plot ready every [interval] step
        '''
        return self.interval % intv == 0
    
    def plot_traj(self):
        '''
        plot obs of each agent and the reference
        '''
        if self._ready():
            for agent in self.agents:  
                plt.plot(self.observation_spaces[agent], "ro")
            plt.plot(self.reference_state_history, "b-") 
            plt.show()    
            
    
        
        