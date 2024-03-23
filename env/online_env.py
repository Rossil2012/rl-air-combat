from utils.tools import make_struct
from env.base_env import norm, denorm
from utils.tcp import connect, recv_exact_n_bytes_into
from utils.define import ACTION_INFO, HEALTH_MAX

import ctypes
import random
import numpy as np

from typing import Iterable, Callable, List
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories.time_step import time_step_spec, transition


_INFO_LEN = 2
_MAX_PLAYER = 6
_ONE_PLAYER_VEC_LEN = 16
_CONTROL_VEC_LEN = 7


class OnlineRecv(ctypes.Structure):
    _fields_ = [
        ('num', ctypes.c_float),
        ('state', ctypes.c_float),
        ('mine', ctypes.c_float * _ONE_PLAYER_VEC_LEN),
        ('opp', ctypes.c_float * _ONE_PLAYER_VEC_LEN)
    ]


class OnlineSend(ctypes.Structure):
    _fields_ = [
        ('pitch', ctypes.c_float),
        ('roll', ctypes.c_float),
        ('yaw', ctypes.c_float),
        ('throttle', ctypes.c_float),
        ('fireType', ctypes.c_float),
        ('IsKeyboardEnabled', ctypes.c_float),
        ('IsJoyStickEnabled', ctypes.c_float),
    ]


def _get_struct_from_buffer(struct_cls, buf):
    p = struct_cls()
    ctypes.memmove(ctypes.addressof(p), buf, ctypes.sizeof(p))
    return p


def _construct_obs(raw_struct, prev_health):
    now_health = [raw_struct.mine[2] * HEALTH_MAX, raw_struct.opp[2] * HEALTH_MAX]
    obs = [*raw_struct.mine[4:], prev_health[0] - now_health[0], now_health[0],
           *raw_struct.opp[4:], prev_health[1] - now_health[1], now_health[1]]
    return np.array(obs), now_health


class OnlineEnv:
    def __init__(self,
                 ip: str, port: int,
                 state_size: int, state_min: Iterable, state_max: Iterable,
                 get_state_fcn: Callable[[np.ndarray], np.ndarray],
                 policy: PyPolicy,
                 policy_info: List = None,
                 action_wrapper: Callable = None
                 ):

        self._ip = ip
        self._port = port
        self._sock = None

        self._state_info = (state_size, state_min, state_max)

        self._get_state_fcn = get_state_fcn
        self._policy = policy
        self._policy_info = policy_info

        self._action_wrapper = action_wrapper or (lambda _: _)

        self._recv_buf = ctypes.create_string_buffer(1024)

        self._health = [HEALTH_MAX] * 2

    def reset(self):
        if self._sock is not None:
            [sock.close() for sock in self._sock]
        self._sock = [connect(self._ip, self._port) for _ in range(2)]
        self._health = [HEALTH_MAX] * 2

    def step(self) -> bool:
        print('before')
        recv_exact_n_bytes_into(self._sock[1], (_INFO_LEN + _MAX_PLAYER * _ONE_PLAYER_VEC_LEN) *
                                ctypes.sizeof(ctypes.c_float), memoryview(self._recv_buf))
        recv_c = _get_struct_from_buffer(OnlineRecv, self._recv_buf)

        print(recv_c)

        if recv_c.state != 0.:
            return True
        else:
            obs, self._health = _construct_obs(recv_c, self._health)
            raw_state = self._get_state_fcn(obs)
            norm_state = norm(raw_state, self._state_info[1], self._state_info[2]).astype(np.float32)
            action = self._policy.action(transition(
                observation=norm_state,
                reward=0.
            )).action

            if self._policy_info is not None:
                print('hierarchy!!!!')
                sub_policy, get_state, state_info, process_action = self._policy_info[action]
                process_action = process_action or (lambda _: _)
                action = process_action(sub_policy.action(transition(
                    observation=norm(get_state(obs), state_info[1], state_info[2]).astype(np.float32),
                    reward=0.
                )).action)

            action = self._action_wrapper(action)
            raw_action = denorm(action, ACTION_INFO[1], ACTION_INFO[2])

            roll, pitch, yaw, throttle = raw_action

            print(obs)
            print(raw_action)

            # roll = 0.
            # pitch = 1000.
            # yaw = 0.
            # throttle = 1000.

            self._sock[1].sendall(make_struct(OnlineSend, pitch, roll, yaw, throttle, 0., 0., 0.))
            # self._sock[1].sendall(make_struct(OnlineSend, roll, pitch, yaw, throttle, 0., 0., 0.))

            return False





























