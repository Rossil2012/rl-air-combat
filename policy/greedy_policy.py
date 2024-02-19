from env.base_env import norm, denorm
from utils.define import ACTION_INFO, COMBAT_OBS_INFO

import math
import numpy as np

from typing import Optional
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.typing.types import Seed, NestedArray
from tf_agents.trajectories.time_step import time_step_spec
from tf_agents.specs.array_spec import BoundedArraySpec, ArraySpec


def _add_bias(num):
    if abs(num) < 1e-4:
        num = 1e-4 if num > 0 else -1e-4
    return num


def get_target_euler_diff(rx, ry, rz, roll, pitch, yaw):
    rx, ry = _add_bias(rx), _add_bias(ry)

    print(rx, ry, rz, roll, pitch, yaw)

    target_pitch = math.atan(rz / math.sqrt(rx ** 2 + ry ** 2))
    pitch_diff = pitch - target_pitch

    target_yaw = math.atan(ry / rx) if rx > 0 else math.atan(ry / rx) + math.pi
    target_yaw = target_yaw if target_yaw > 0 else 2 * math.pi + target_yaw
    yaw_diff = yaw - target_yaw
    if yaw_diff < -math.pi:
        yaw_diff = 2 * math.pi + yaw_diff
    elif yaw_diff > math.pi:
        yaw_diff = 2 * math.pi - yaw_diff

    return 0., pitch_diff, yaw_diff


class GreedyPolicy(PyPolicy):
    """
    通过计算当前欧拉角与预期欧拉角控制转向，根据距离控制油门
    """
    def __init__(self):
        ts_spec, = time_step_spec(
            BoundedArraySpec((COMBAT_OBS_INFO[0],), np.float32,
                             minimum=COMBAT_OBS_INFO[1], maximum=COMBAT_OBS_INFO[2], name='observation'),
            ArraySpec(shape=(), dtype=np.float32, name='reward')
        ),
        action_spec = BoundedArraySpec((ACTION_INFO[0],), np.float32,
                                       minimum=ACTION_INFO[0]*[-1], maximum=ACTION_INFO[0]*[1], name='action')
        super(GreedyPolicy, self).__init__(
            time_step_spec=ts_spec,
            action_spec=action_spec
        )

    def _action(self,
                time_step: TimeStep,
                policy_state: NestedArray,
                seed: Optional[Seed] = None) -> PolicyStep:

        # 拿到的obs是归一化过的，需要还原
        obs = denorm(time_step.observation, COMBAT_OBS_INFO[1], COMBAT_OBS_INFO[2])
        rx, ry, rz, roll, pitch, yaw = obs[14] - obs[0], obs[15] - obs[1], obs[16] - obs[2], obs[3], obs[4], obs[5]

        dist = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

        roll_diff, pitch_diff, yaw_diff = get_target_euler_diff(rx, ry, rz, roll, pitch, yaw)

        print('greedy:')
        print(rx, ry, rz, pitch, yaw)
        print(roll_diff, pitch_diff, yaw_diff)

        # 限制了转弯的速度和油门
        pitch_action = 400. if pitch_diff < 0. else -400.
        yaw_action = 400. if yaw_diff < 0. else -400.
        throttle_action = 1000. if dist > 300. else 0.

        # 输出的动作需要归一化
        norm_action = norm(np.array([0., pitch_action, yaw_action, throttle_action]), ACTION_INFO[1], ACTION_INFO[2])

        return PolicyStep(norm_action, None, None)
