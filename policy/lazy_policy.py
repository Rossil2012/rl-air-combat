from env.base_env import norm, denorm
from utils.define import ACTION_INFO, COMBAT_OBS_INFO

import numpy as np

from typing import Optional
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.typing.types import Seed, NestedArray
from tf_agents.trajectories.time_step import time_step_spec
from tf_agents.specs.array_spec import BoundedArraySpec, ArraySpec


class LazyPolicy(PyPolicy):
    """
    战机不做任何动作，所有控制量均为0
    """
    def __init__(self):
        ts_spec, = time_step_spec(
            BoundedArraySpec((COMBAT_OBS_INFO[0],), np.float32,
                             minimum=COMBAT_OBS_INFO[1], maximum=COMBAT_OBS_INFO[2], name='observation'),
            ArraySpec(shape=(), dtype=np.float32, name='reward')
        ),
        action_spec = BoundedArraySpec((ACTION_INFO[0],), np.float32,
                                       minimum=ACTION_INFO[0]*[-1], maximum=ACTION_INFO[0]*[1], name='action')
        super(LazyPolicy, self).__init__(
            time_step_spec=ts_spec,
            action_spec=action_spec
        )

    def _action(self,
                time_step: TimeStep,
                policy_state: NestedArray,
                seed: Optional[Seed] = None) -> PolicyStep:

        # 拿到的obs是归一化过的，需要还原
        obs = denorm(time_step.observation, COMBAT_OBS_INFO[1], COMBAT_OBS_INFO[2])

        roll_action = 0.
        pitch_action = 0.
        yaw_action = 0.
        throttle_action = 0.

        # 输出的动作需要归一化
        norm_action = norm(np.array([roll_action, pitch_action, yaw_action, throttle_action]), ACTION_INFO[1], ACTION_INFO[2])

        return PolicyStep(norm_action, None, None)
