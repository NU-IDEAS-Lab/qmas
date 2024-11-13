# This file sets up the patrolling environment wrappers.

from functools import wraps
from .communication.communication_wrapper import CommunicationWrapper
from .communication.model.model_boolean_observation import BooleanObservationModel

from patrolling_zoo.env.patrolling_zoo import parallel_env as base_env

# We import these functions so that they are found by the Pettingzoo trainer.
from patrolling_zoo.env.patrolling_zoo import add_args, validate_args

# We use the @wraps decorator to ensure that Python introspection tools used by Pettingzoo config work on the base class, not the wrapper function.
@wraps(base_env)
def env(*args, **kwargs):
    ''' Creates the environment with the specified wrappers. '''

    comms_model = BooleanObservationModel()

    environment = base_env(*args, **kwargs)
    environment = CommunicationWrapper(environment, comms_model)

    return environment