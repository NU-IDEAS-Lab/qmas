import numpy as np
import torch
import os.path
from onpolicy.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy
from .actor_critic import QmasActor, QmasCritic
from .predictor import Predictor

from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


class QmasPolicy(R_MAPPOPolicy):
    ''' This class implements the QMAS policy, including communication. '''

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.args = args

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # We use the QMAS actor, but the default MAPPO critic.
        self.actor = QmasActor(args, self.obs_space, self.act_space, self.device)
        self.critic = QmasCritic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        
        # Create the predictor / diffusion model.
        obs_shape = get_shape_from_obs_space(self.obs_space, flatten_dicts=False)
        obs_dim = np.prod(obs_shape) # observation space for one agent
        action_dim = np.prod(get_shape_from_act_space(act_space)) # action space for one agent

        if args.prediction_ensemble_size > 1:
            print(f"Creating ensemble of {args.prediction_ensemble_size} predictors.")
        
        self.predictors = []
        for i in range(args.prediction_ensemble_size):
            if args.cuda and torch.cuda.is_available():
                if len(args.cuda_idx_predictor) > 1:
                    device_predictor = torch.device(f"cuda:{args.cuda_idx_predictor[i]}")
                elif len(args.cuda_idx_predictor) == 1:
                    device_predictor = torch.device(f"cuda:{args.cuda_idx_predictor[0]}")
                else:
                    device_predictor = self.device
            else:
                device_predictor = self.device

            predictor = Predictor(
                obs_dim,
                action_dim,
                args,
                obs_shape=obs_shape,
                device=device_predictor
            ).to(device_predictor)
            self.predictors.append(predictor)


    def get_prediction(self, trajectory, visibility_mask=None, prediction_prev=None, return_member_preds=False):
        """
        Get a prediction from the ensemble of predictors.
        Args:
            trajectory: A tensor of shape (T, D), where T is the trajectory length and D is the transition dimension (action + observation).
            visibility_mask: An optional tensor of shape (T,) indicating which timesteps are visible (1) or not (0).
            prediction_prev: An optional tensor of shape (T, D_out) representing the previous prediction to condition on.
        Returns:
            prediction: A tensor of shape (T, D_out) representing the mean prediction across the ensemble.
            uncertainty: A tensor of shape (T, D_out) representing the uncertainty (variance) across the ensemble predictions.
        """

        predictions = []
        for i, predictor in enumerate(self.predictors):
            member_prev = None if prediction_prev is None else prediction_prev[i].to(predictor.device)
            pred = predictor.get_prediction(
                trajectory.clone().to(predictor.device),
                visibility_mask.clone().to(predictor.device),
                member_prev)
            predictions.append(pred.unsqueeze(0).to(self.device))
        predictions = torch.cat(predictions, dim=0)  # Shape: (num_predictors, 1, T, D_out)
        prediction = predictions.mean(dim=0)
        if predictions.shape[0] > 1:
            uncertainty = predictions.std(dim=0)
        else:
            uncertainty = torch.zeros_like(prediction)

        if return_member_preds:
            return prediction, uncertainty, predictions.detach()
        return prediction, uncertainty


    def save(self, directory, episode):
        ''' Save the policy. '''

        super().save(directory, episode)

        for i, predictor in enumerate(self.predictors):
            torch.save(predictor.state_dict(), os.path.join(directory, f"predictor{i}.pt"))


    # Predictor tensors whose shape is a deterministic function of the current
    # obs space (T x D_flat) rather than learned content. On restore we keep
    # the freshly-constructed values and ignore whatever the checkpoint stored,
    # so a model trained with one num_agents can be evaluated with another.
    # Learned weights (lin_in_X, lin_out_X, etc.) are F-sized and load normally.
    _LAYOUT_TIED_PREDICTOR_KEYS = (
        "diffuser.fix_mask",
        "diffuser.loss_weight",
        "gnn_edge_fix_template",
        "obs_running_mean",
        "obs_running_var",
        "obs_running_count",
    )

    def restore(self, directory):
        ''' Restore the policy. '''

        super().restore(directory)

        for i, predictor in enumerate(self.predictors):
            predictor_state_dict = torch.load(os.path.join(directory, f"predictor{i}.pt"), map_location=self.device)

            current_state = predictor.state_dict()
            for key in self._LAYOUT_TIED_PREDICTOR_KEYS:
                if key in predictor_state_dict and key in current_state:
                    predictor_state_dict[key] = current_state[key]
            # diffuser.fix_mask is popped out of _parameters at construction
            # so it doesn't appear in current_state; old checkpoints may still
            # carry it. Pull from the live attribute when that happens.
            if (
                "diffuser.fix_mask" in predictor_state_dict
                and "diffuser.fix_mask" not in current_state
            ):
                predictor_state_dict["diffuser.fix_mask"] = predictor.diffuser.fix_mask

            predictor.load_state_dict(predictor_state_dict)