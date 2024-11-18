from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO

import numpy as np

class QmasAlgorithm(R_MAPPO):
    """
    Trainer class for QMAS to update policies.
    """

    # The two-module variant actually uses the same algorithm as regular MAPPO. We just need to update the Actor class.
    pass
