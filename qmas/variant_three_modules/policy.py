import torch
from onpolicy.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy

class QmasPolicy(R_MAPPOPolicy):
    ''' This class implements the QMAS policy, including communication. '''

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        ''' Initializes the QMAS policy. '''
        super().__init__(args, obs_space, cent_obs_space, act_space, device)

        # Initialize the communication module.
        self.comms = torch.nn.Linear(args.hidden_size, args.num_agents)
        self.comms_optimizer = torch.optim.Adam(self.comms.parameters(),
                                               lr=args.lr, eps=args.opti_eps,
                                               weight_decay=args.weight_decay)
        
        print("Initialized QMAS policy.")
    
    def comms_update(sample):
        ''' Updates the communication module. '''
        
        raise NotImplementedError("Communication update not implemented.")