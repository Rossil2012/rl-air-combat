from utils.define import ACTION_INFO, COMBAT_OBS_INFO

import functools
import numpy as np

from tf_agents.trajectories.time_step import time_step_spec
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.specs.array_spec import BoundedArraySpec, ArraySpec

RandomPolicy = functools.partial(
    RandomPyPolicy,
    time_step_spec=time_step_spec(
            BoundedArraySpec((COMBAT_OBS_INFO[0],), np.float32, minimum=COMBAT_OBS_INFO[1], maximum=COMBAT_OBS_INFO[2],
                             name='observation'),
            ArraySpec(shape=(), dtype=np.float32, name='reward')
        ),
    action_spec=BoundedArraySpec((ACTION_INFO[0],), np.float32,
                                 minimum=ACTION_INFO[0]*[-1], maximum=ACTION_INFO[0]*[1], name='action')
)
