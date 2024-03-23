from utils.tools import make_struct
from env.base_env import norm, denorm
from utils.tcp import connect, recv_exact_n_bytes_into
from utils.define import PI, WebpackHeader, Posture, TrainingModeInitInfo, TrainingModeProcessInfo, FlightControl, \
    WeaponControl, DaotiaoType, OperationType, FireType, INIT_CALLBACK_ID, PROCESS_CALLBACK_ID, RESTART_CALLBACK_ID, \
    ACTION_INFO, COMBAT_OBS_INFO, HEALTH_MAX

import ctypes
import random
import numpy as np

from typing import Iterable, Callable, List
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories.time_step import time_step_spec, transition


def _gen_random_init_pos():
    return [random.uniform(COMBAT_OBS_INFO[1][i], COMBAT_OBS_INFO[2][i]) for i in range(6)]


def _extract_fields(instance):
    ret = []
    for field in instance._fields_:
        content = instance.__getattribute__(field[0])
        if isinstance(content, ctypes.Array):
            ret.extend(content)
        else:
            ret.append(content)
    return ret


def _get_struct_from_buffer(struct_cls, buf):
    p = struct_cls()
    ctypes.memmove(ctypes.addressof(p), buf, ctypes.sizeof(p))
    return p


def _construct_obs(process_info, prev_health):
    obs = []
    for info in process_info:
        post = info.self
        obs += _extract_fields(post)
        raw_health = HEALTH_MAX * info.m_HPNormalized
        obs += [raw_health - prev_health, raw_health]
    return np.array(obs)


_WPH_INIT_ARGS = [OperationType.INI_POINT.value, DaotiaoType.NONE.value, 11112222,
                  ctypes.sizeof(TrainingModeInitInfo), INIT_CALLBACK_ID]
_WPH_PROCESS_ARGS = [OperationType.EPOCH_CLIENT_TO_COMM.value, DaotiaoType.NONE.value, 0,
                     ctypes.sizeof(FlightControl) + ctypes.sizeof(WeaponControl), PROCESS_CALLBACK_ID]
_WPH_RESTART_ARGS = [OperationType.TRAINING_RESTART.value, DaotiaoType.NONE.value, 11112222, 0, RESTART_CALLBACK_ID]


class DisplayEnv:
    def __init__(self,
                 ip: List[str], port: List[int],
                 state_size: List[int], state_min: List[Iterable], state_max: List[Iterable],
                 get_state_fcn: List[Callable[[np.ndarray], np.ndarray]],
                 policies: List[PyPolicy],
                 policy_info: List = None,
                 action_wrappers: List[Callable] = None,
                 gen_init_pos_fcn: Callable[[], List] = None,
                 max_step: int = 10000
                 ):

        self._ip = ip
        self._port = port
        self._sock = None

        self._state_info = [(state_size[i], state_min[i], state_max[i]) for i in range(2)]

        self._get_state_fcn = get_state_fcn
        self._policies = policies
        self._policy_info = policy_info or [None, None]
        self._conn_num = len(policies)
        assert self._conn_num == 2

        action_wrappers = action_wrappers or [lambda _: _] * 2
        self._action_wrappers = [(wrapper or (lambda _:_)) for wrapper in action_wrappers]

        self._gen_init_pos_fcn = gen_init_pos_fcn or _gen_random_init_pos
        self._recv_buf = ctypes.create_string_buffer(1024)

        self._health = [HEALTH_MAX] * self._conn_num
        self._cur_step = 0
        self._max_step = max_step
        self._running = False

    def _close_sockets(self):
        if self._sock is not None:
            for sock in self._sock:
                sock.close()

    def force_restart(self):
        if self._running:
            self._sock[0].sendall(make_struct(WebpackHeader, *_WPH_RESTART_ARGS))
            self._close_sockets()
            self._running = False

    def reset(self):
        self._sock = [connect(self._ip[i], self._port[i]) for i in range(self._conn_num)]
        self._health = [HEALTH_MAX] * self._conn_num

        for i, sock in enumerate(self._sock):
            sock.sendall(make_struct(WebpackHeader, *_WPH_INIT_ARGS))
            sock.sendall(make_struct(TrainingModeInitInfo, i+1, *(self._gen_init_pos_fcn() + [0.] * 6)))

        self._cur_step = 0
        self._running = True

    def step(self) -> bool:
        self._cur_step += 1
        if self._cur_step > self._max_step:
            print('到达最大步长限制（{}），没有战机被击落，游戏结束！'.format(self._max_step))
            self.force_restart()
            return True

        recv_exact_n_bytes_into(self._sock[0], ctypes.sizeof(WebpackHeader), memoryview(self._recv_buf))
        header = _get_struct_from_buffer(WebpackHeader, self._recv_buf)
        recv_exact_n_bytes_into(self._sock[1], ctypes.sizeof(WebpackHeader), memoryview(self._recv_buf))

        if header.operationType == OperationType.EPOCH_COMM_TO_CLIENT.value:
            for i in range(self._conn_num):
                process_info = []
                for j in range(self._conn_num):
                    recv_exact_n_bytes_into(self._sock[i], ctypes.sizeof(TrainingModeProcessInfo),
                                            memoryview(self._recv_buf))
                    process_info.append(_get_struct_from_buffer(TrainingModeProcessInfo, self._recv_buf))
                _WPH_PROCESS_ARGS[2] = header.timestamp
                self._sock[i].sendall(make_struct(WebpackHeader, *_WPH_PROCESS_ARGS))

                if i == 1:
                    process_info.reverse()
                obs = _construct_obs(process_info, self._health[i])
                print(obs)
                self._health[i] = HEALTH_MAX * process_info[0].m_HPNormalized
                raw_state = self._get_state_fcn[i](obs)
                norm_state = norm(raw_state, self._state_info[i][1], self._state_info[i][2]).astype(np.float32)
                action = self._policies[i].action(transition(
                    observation=norm_state,
                    reward=0.
                )).action

                if self._policy_info[i] is not None:
                    sub_policy, get_state, state_info, process_action = self._policy_info[i][action]
                    process_action = process_action or (lambda _: _)
                    action = process_action(sub_policy.action(transition(
                        observation=norm(get_state(obs), state_info[1], state_info[2]).astype(np.float32),
                        reward=0.
                    )).action)

                action = self._action_wrappers[i](action)
                raw_action = denorm(action, ACTION_INFO[1], ACTION_INFO[2])
                print('raw_action', raw_action)
                self._sock[i].sendall(
                    make_struct(FlightControl, raw_action[1], raw_action[0], raw_action[2], raw_action[3]))
                self._sock[i].sendall(make_struct(WeaponControl, FireType.GUN.value))

            print('步数：{}， 战机1剩余生命值：{}， 战机2剩余生命值：{}'.format(self._cur_step, *self._health))
            return False
        elif header.operationType == OperationType.GAME_OVER.value:
            recv_exact_n_bytes_into(self._sock[0], header.contentLength, memoryview(self._recv_buf))
            winner = ctypes.cast(self._recv_buf, ctypes.POINTER(ctypes.c_int32)).contents.value
            print('战机{}胜利！'.format(winner))
            self._close_sockets()
            self._running = False
            return True


















































