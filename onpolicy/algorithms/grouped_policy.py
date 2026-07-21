"""Per-agent-class (grouped) policies.

The shared runners give every agent one set of actor weights; the ``separated``
runners give every agent its own. Neither fits an environment whose agents fall
into a handful of *classes* (ISRU: prospector / extractor / hauler / ...), where
we want weight sharing within a class but distinct weights across classes.

``make_grouped_policy`` wraps any existing policy class so that it holds one
actor per group while keeping a single centralized critic and a single predictor
ensemble:

    GroupedPolicy
     |- actor: GroupedActor
     |    `- actors: ModuleList[BaseActor x n_groups]   <- separated
     |- critic:     <- shared (centralized-V emits one value per rollout thread)
     `- predictors: <- shared (a world model, not an agent behaviour model)

With a single group this is behaviourally identical to the unwrapped policy.
"""

import inspect

import numpy as np
import torch
import torch.nn as nn


def _build_actor(actor_cls, args, obs_space, share_obs_space, act_space, device):
    """Instantiate an actor of the same type as the base policy's actor.

    Actor constructors in this repo come in two shapes: the newer
    ``(args, obs_space, share_obs_space, action_space, device)`` used by
    ``qmas.variant_two_modules.new.actor_critic.QmasActor`` and the older
    ``(args, obs_space, action_space, device)``. Pick by signature rather than
    guessing, and fail loudly on anything else so a mismatch surfaces here
    instead of as a confusing shape error deep in the forward pass.
    """
    params = list(inspect.signature(actor_cls.__init__).parameters)[1:]  # drop self
    positional = [p for p in params if p != "device"]

    if len(positional) == 4:
        return actor_cls(args, obs_space, share_obs_space, act_space, device=device)
    if len(positional) == 3:
        return actor_cls(args, obs_space, act_space, device=device)
    raise TypeError(
        f"Cannot build an extra actor of type {actor_cls.__name__}: unsupported "
        f"constructor signature {params}. Grouped policies expect either "
        f"(args, obs_space, share_obs_space, action_space, device) or "
        f"(args, obs_space, action_space, device)."
    )


def _take(x, idx_np, idx_torch):
    """Index a batch-major input, tolerating None / object arrays / tensors."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x[idx_np]
    if isinstance(x, torch.Tensor):
        return x[idx_torch.to(x.device)]
    # Lists and other sequences (rare, but graph obs sometimes arrive as lists).
    return [x[i] for i in idx_np]


class GroupedActor(nn.Module):
    """Dispatches each row of a batch to the actor owning that agent's group.

    Row -> group resolution is deliberately explicit. Batch row order is *not*
    the same everywhere: rollouts and ``feed_forward_generator`` produce
    ``(threads, agents)`` row-major order, but ``recurrent_generator`` emits
    chunks keyed by ``(thread, agent)`` pairs, so ``row % num_agents`` is wrong
    there. Callers that know their ordering pass ``agent_ids``; only the
    row-major case is inferred.
    """

    def __init__(self, actors, agent_group_ids, num_agents):
        super().__init__()
        self.actors = nn.ModuleList(actors)
        self.num_agents = num_agents
        self.register_buffer(
            "agent_group_ids",
            torch.as_tensor(np.asarray(agent_group_ids), dtype=torch.long),
            persistent=False,
        )
        # The runner reads `policy.actor.act.log_prob_dim` to size the log-prob
        # buffer. The distribution head is structurally identical across groups.
        self.act = self.actors[0].act

    @property
    def n_groups(self):
        return len(self.actors)

    def _row_groups(self, batch_size, agent_ids):
        """Return a LongTensor of length batch_size giving each row's group."""
        if agent_ids is not None:
            agent_ids = torch.as_tensor(np.asarray(agent_ids), dtype=torch.long).reshape(-1)
            if agent_ids.numel() != batch_size:
                raise ValueError(
                    f"agent_ids has {agent_ids.numel()} entries but the batch has "
                    f"{batch_size} rows."
                )
            return self.agent_group_ids.cpu()[agent_ids]

        # Inferred row-major (threads, agents) ordering.
        if batch_size % self.num_agents != 0:
            raise ValueError(
                f"Cannot infer agent ids: batch size {batch_size} is not a multiple "
                f"of num_agents {self.num_agents}. Pass agent_ids explicitly."
            )
        reps = batch_size // self.num_agents
        return self.agent_group_ids.cpu().repeat(reps)

    def _group_slices(self, batch_size, agent_ids):
        """Yield (actor, idx_np, idx_torch) for each non-empty group."""
        row_groups = self._row_groups(batch_size, agent_ids)
        for g, actor in enumerate(self.actors):
            idx_torch = torch.nonzero(row_groups == g, as_tuple=False).reshape(-1)
            if idx_torch.numel() == 0:
                continue
            yield actor, idx_torch.numpy(), idx_torch

    @staticmethod
    def _scatter(pieces, batch_size):
        """Reassemble per-group outputs into original row order."""
        first = pieces[0][1]
        out = torch.empty(
            (batch_size, *first.shape[1:]), dtype=first.dtype, device=first.device
        )
        for idx_torch, values in pieces:
            out[idx_torch.to(out.device)] = values.to(out.device)
        return out

    def forward(self, obs, rnn_states, masks, available_actions=None,
                deterministic=False, global_obs=None, agent_ids=None):
        batch_size = len(obs)

        actions, log_probs, states = [], [], []
        for actor, idx_np, idx_torch in self._group_slices(batch_size, agent_ids):
            a, lp, rs = actor(
                _take(obs, idx_np, idx_torch),
                _take(rnn_states, idx_np, idx_torch),
                _take(masks, idx_np, idx_torch),
                _take(available_actions, idx_np, idx_torch),
                deterministic,
                global_obs=_take(global_obs, idx_np, idx_torch),
            )
            actions.append((idx_torch, a))
            log_probs.append((idx_torch, lp))
            states.append((idx_torch, rs))

        return (
            self._scatter(actions, batch_size),
            self._scatter(log_probs, batch_size),
            self._scatter(states, batch_size),
        )

    def evaluate_actions(self, obs, global_obs, rnn_states, action, masks,
                         available_actions=None, active_masks=None, agent_ids=None):
        batch_size = len(obs)

        log_probs = []
        entropy_terms = []  # (row_count, scalar entropy) for a weighted mean
        for actor, idx_np, idx_torch in self._group_slices(batch_size, agent_ids):
            lp, ent = actor.evaluate_actions(
                _take(obs, idx_np, idx_torch),
                _take(global_obs, idx_np, idx_torch),
                _take(rnn_states, idx_np, idx_torch),
                _take(action, idx_np, idx_torch),
                _take(masks, idx_np, idx_torch),
                _take(available_actions, idx_np, idx_torch),
                _take(active_masks, idx_np, idx_torch),
            )
            log_probs.append((idx_torch, lp))
            entropy_terms.append((idx_torch.numel(), ent))

        # dist_entropy is a batch mean, so recombining groups needs to be weighted
        # by row count. An unweighted mean would silently reweight the entropy
        # bonus by group size and break the single-group no-op guarantee.
        total_rows = sum(n for n, _ in entropy_terms)
        dist_entropy = sum(ent * (n / total_rows) for n, ent in entropy_terms)

        return self._scatter(log_probs, batch_size), dist_entropy


def make_grouped_policy(base_policy_cls):
    """Return a subclass of `base_policy_cls` with one actor per agent group."""

    class GroupedPolicy(base_policy_cls):
        __doc__ = (
            f"{base_policy_cls.__name__} with one actor per agent group. "
            "The critic and predictor ensemble remain shared."
        )

        def __init__(self, args, obs_space, cent_obs_space, act_space,
                     device=torch.device("cpu"), **kwargs):
            # Builds the shared critic, the shared predictors, and actor #0.
            super().__init__(args, obs_space, cent_obs_space, act_space,
                             device=device, **kwargs)

            group_ids = getattr(args, "agent_group_ids", None)
            if group_ids is None:
                group_ids = np.zeros(args.num_agents, dtype=np.int64)
            group_ids = np.asarray(group_ids, dtype=np.int64)

            if len(group_ids) != args.num_agents:
                raise ValueError(
                    f"agent_group_ids has {len(group_ids)} entries but there are "
                    f"{args.num_agents} agents."
                )

            n_groups = int(group_ids.max()) + 1 if len(group_ids) else 1
            base_actor = self.actor
            actors = [base_actor] + [
                _build_actor(type(base_actor), args, self.obs_space,
                             self.share_obs_space, self.act_space, device)
                for _ in range(n_groups - 1)
            ]

            self.actor = GroupedActor(actors, group_ids, args.num_agents).to(device)

            # One optimizer over every group's parameters, matching the base
            # policy's hyperparameters.
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(),
                lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay,
            )

            print(
                f"GroupedPolicy: {n_groups} actor group(s) over {args.num_agents} "
                f"agents (agent -> group: {group_ids.tolist()})"
            )

        def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                        available_actions=None, deterministic=False, global_obs=None,
                        agent_ids=None):
            actions, action_log_probs, rnn_states_actor = self.actor(
                obs, rnn_states_actor, masks, available_actions, deterministic,
                global_obs=global_obs, agent_ids=agent_ids,
            )

            # Critic handling is unchanged from the base policy: centralized-V
            # produces one value per thread, so rnn/masks are sliced per thread.
            n_agents = self.args.num_agents
            if rnn_states_critic.shape[0] == cent_obs.shape[0] * n_agents:
                values, updated_rnn = self.critic(
                    cent_obs, rnn_states_critic[::n_agents], masks[::n_agents]
                )
                rnn_states_critic = updated_rnn.repeat_interleave(n_agents, dim=0)
            else:
                values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)

            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

        def act(self, obs, rnn_states_actor, masks, available_actions=None,
                deterministic=False, global_obs=None, agent_ids=None):
            actions, _, rnn_states_actor = self.actor(
                obs, rnn_states_actor, masks, available_actions, deterministic,
                global_obs=global_obs, agent_ids=agent_ids,
            )
            return actions, rnn_states_actor

        def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic,
                             action, masks, available_actions=None, active_masks=None,
                             global_obs=None, agent_ids=None):
            action_log_probs, dist_entropy = self.actor.evaluate_actions(
                obs, global_obs, rnn_states_actor, action, masks,
                available_actions, active_masks, agent_ids=agent_ids,
            )
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
            return values, action_log_probs, dist_entropy

        def restore(self, directory):
            try:
                super().restore(directory)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to restore a grouped policy from {directory}. Grouped "
                    f"checkpoints store actors under 'actors.<i>.*' and are not "
                    f"interchangeable with shared-policy checkpoints. Original "
                    f"error: {e}"
                ) from e

    GroupedPolicy.__name__ = f"Grouped{base_policy_cls.__name__}"
    GroupedPolicy.__qualname__ = GroupedPolicy.__name__
    return GroupedPolicy
