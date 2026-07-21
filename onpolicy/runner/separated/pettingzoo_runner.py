import importlib
import sys

from onpolicy.algorithms.grouped_policy import make_grouped_policy
from onpolicy.runner.shared.pettingzoo_runner import PettingzooRunner as _SharedPettingzooRunner


# Grouped subclasses are registered into this module so the base runner's
# importlib-based `policy_class` resolution can find them by dotted path.
_GROUPED_CACHE = {}


def _as_grouped(policy_class_path):
    """Wrap a dotted `module.ClassName` policy path in a grouped equivalent.

    Returns a dotted path pointing at a generated subclass registered in this
    module, so it resolves through the same importlib lookup the shared base
    runner already uses rather than bypassing it.
    """
    if not policy_class_path:
        # Empty means the base runner falls back to R_MAPPOPolicy.
        from onpolicy.algorithms.r_mappo.rMAPPOPolicy import R_MAPPOPolicy as base_cls
    else:
        module_name, class_name = policy_class_path.rsplit(".", 1)
        try:
            base_cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Invalid policy module: {policy_class_path}. Import failed.") from e

    if base_cls.__name__ not in _GROUPED_CACHE:
        grouped_cls = make_grouped_policy(base_cls)
        _GROUPED_CACHE[base_cls.__name__] = grouped_cls
        setattr(sys.modules[__name__], grouped_cls.__name__, grouped_cls)

    grouped_cls = _GROUPED_CACHE[base_cls.__name__]
    return f"{__name__}.{grouped_cls.__name__}"


class PettingzooRunner(_SharedPettingzooRunner):
    """PettingZoo runner with one actor per agent class.

    Agents of the same class share actor weights; different classes get
    different weights. The centralized critic and the predictor ensemble stay
    shared, matching how they are defined: the critic emits one value per
    rollout thread over the global state, and the predictor is a world model
    rather than a model of any particular agent's behaviour.

    Everything else — rollout collection, the trajectory buffer, UQ injection,
    Zarr eval, rendering — is inherited unchanged from the shared runner.
    """

    def __init__(self, config):
        all_args = config["all_args"]

        agent_group_ids = getattr(config["envs"], "agent_group_ids", None)
        if agent_group_ids is None:
            raise ValueError(
                "The environment did not report agent classes, so per-class policies "
                "cannot be built. Expose `agent_class_names` on the environment, or "
                "run with --share_policy."
            )
        # Store as a plain list: all_args is yaml-dumped into the run's config.yaml,
        # and safe_dump cannot represent a numpy array.
        all_args.agent_group_ids = [int(g) for g in agent_group_ids]
        all_args.policy_class = _as_grouped(all_args.policy_class)

        super().__init__(config)
