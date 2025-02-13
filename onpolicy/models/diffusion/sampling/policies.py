from collections import namedtuple
import torch
import einops

# from diffuser.datasets.preprocessing import get_policy_preprocess_fn
from ..helpers import to_np, to_torch, apply_dict


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:

    # def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
    def __init__(self, guide, diffusion_model, normalizer=None, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        # self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        # conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        # conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = to_np(samples.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        if self.normalizer is not None:
            actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        if self.normalizer is not None:
            observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        if self.normalizer is not None:
            conditions = apply_dict(
                self.normalizer.normalize,
                conditions,
                'observations',
            )
        conditions = to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
