from env.base_env import norm, denorm
from env.socket_env import SocketEnv
from utils.define import PI, KineSendStruct, KineRecvStruct, KineStateInitStruct, \
    ACTION_INFO, COMBAT_OBS_INFO, NO_GUN_OBS_INFO, HEALTH_MAX, DAMAGE_MAX, DAMAGE_DIST_MAX, DAMAGE_ANGLE_MAX


import math
import random
import numpy as np

from typing import Tuple, Callable, Optional, Any, List
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.specs.array_spec import BoundedArraySpec, ArraySpec
from tf_agents.trajectories.time_step import time_step_spec, transition


COMBAT_PLAYS = 2


class CombatEnv(SocketEnv):
    def __init__(self,
                 ip: str, port: int,
                 mock_policy_fcn: Callable[[], PyPolicy],
                 state_size: int, state_min: List, state_max: List,
                 get_state_fcn: Callable[[np.ndarray], np.ndarray],
                 get_reward_fcn: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[float, bool]],
                 gen_init_pos_fcn: Callable[[], List] = None,
                 mock_policy_info: List = None,
                 max_step: int = 1000,
                 introduce_damage: bool = False
                 ):

        self._get_state_fcn = get_state_fcn
        self._get_reward_fcn = get_reward_fcn

        self._introduce_damage = introduce_damage
        self._health = self._health_mock = HEALTH_MAX

        self._gen_init_pos_fcn = gen_init_pos_fcn or self._gen_random_init_pos

        self._action_info = ACTION_INFO
        self._state_info = state_size, state_min, state_max

        self._obs_info = COMBAT_OBS_INFO if introduce_damage else NO_GUN_OBS_INFO

        self._mock_policy = mock_policy_fcn()
        self._mock_policy_info = mock_policy_info or ([lambda _: _] * 2 + list(COMBAT_OBS_INFO[1:3]))

        super(CombatEnv, self).__init__(
            ip=[ip] * COMBAT_PLAYS, port=[port] * COMBAT_PLAYS,
            init_struct=[KineStateInitStruct] * COMBAT_PLAYS,
            send_struct=[KineSendStruct] * COMBAT_PLAYS,
            recv_struct=[KineRecvStruct] * COMBAT_PLAYS,
            action_size=[self._action_info[0]] * COMBAT_PLAYS,
            action_min=[self._action_info[1]] * COMBAT_PLAYS,
            action_max=[self._action_info[2]] * COMBAT_PLAYS,
            state_size=[state_size] * COMBAT_PLAYS,
            state_min=[state_min] * COMBAT_PLAYS,
            state_max=[state_max] * COMBAT_PLAYS,
            max_step=max_step,
        )

    @property
    def all_observations(self):
        return self._all_observations

    def _gen_random_init_pos(self):
        return [[random.uniform(self._obs_info[1][i], self._obs_info[2][i]) for i in range(6)] for _ in range(2)]

    def _gen_concat_obs(self, obs, obs_mock):
        def _cal_damage(rx, ry, rz, pitch, yaw):
            dist = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
            vec_head = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
            vec_point = np.array([rx, ry, rz])
            dot_res = np.dot(vec_head, vec_point)
            norm_res = np.linalg.norm(vec_head) * np.linalg.norm(vec_point)
            angle_diff = np.pi if norm_res == 0. else np.arccos(max(min(dot_res / norm_res, 1.), -1.))

            if angle_diff < DAMAGE_ANGLE_MAX and dist < DAMAGE_DIST_MAX:
                return DAMAGE_MAX * (1 - dist / DAMAGE_DIST_MAX)
            else:
                return 0.

        if self._introduce_damage:
            rx, ry, rz = obs_mock[0] - obs[0], obs_mock[1] - obs[1], obs_mock[2] - obs[2]
            damage_to_mock = _cal_damage(rx, ry, rz, obs[4], obs[5])
            damage_from_mock = _cal_damage(-rx, -ry, -rz, obs_mock[4], obs_mock[5])
            self._health = max(self._health - damage_from_mock, 0.)
            self._health_mock = max(self._health_mock - damage_to_mock, 0.)
            obs += [damage_from_mock, self._health]
            obs_mock += [damage_to_mock, self._health_mock]

        concat_obs = np.array(obs + obs_mock, dtype=np.float32)
        concat_obs_mock = np.array(obs_mock + obs, dtype=np.float32)

        return concat_obs, concat_obs_mock

    def _get_state(self, obs: np.ndarray) -> np.ndarray:
        return self._get_state_fcn(obs)

    def _get_reward(self, obs: np.ndarray, prev_obs: Optional[np.ndarray]) -> Tuple[float, bool]:
        return self._get_reward_fcn(obs, prev_obs)

    def _init_phase(self):
        self._health = self._health_mock = HEALTH_MAX

    def _get_init_content(self) -> Tuple[List, List]:
        init_to_send = self._gen_init_pos_fcn()
        init_all_obs = [init_to_send[i] + [0.] * 6 for i in range(COMBAT_PLAYS)]
        return init_to_send, init_all_obs

    def _process_observations(self, all_obs: List[List]) -> List[np.ndarray]:
        return [*self._gen_concat_obs(all_obs[0], all_obs[1])]

    def _gen_actions(self, all_observations: List[np.ndarray]) -> List[np.ndarray]:
        get_state, process_action, state_min, state_max = self._mock_policy_info
        raw_state = get_state(all_observations[0])
        norm_state = norm(raw_state, state_min, state_max).astype(np.float32)
        norm_action = process_action(self._mock_policy.action(transition(observation=norm_state, reward=0.)).action)
        return [denorm(norm_action, ACTION_INFO[1], ACTION_INFO[2])]
