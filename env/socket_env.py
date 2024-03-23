from env.base_env import BaseEnv
from utils.tcp import connect, recv_exact_n_bytes_into, struct_pack_into, struct_unpack_from

import abc
import ctypes
import numpy as np

from typing import Tuple, Iterable, Optional, List, Callable


class SocketEnv(BaseEnv):
    def __init__(self,
                 ip: List[str], port: List[int],
                 init_struct: List[Callable],
                 send_struct: List[Callable],
                 recv_struct: List[Callable],
                 action_size: List[int], action_min: List[Iterable], action_max: List[Iterable],
                 state_size: List[int], state_min: List[Iterable], state_max: List[Iterable],
                 max_step: int = 1000
                 ):

        self._ip = ip
        self._port = port

        self._action_info = (action_size, action_min, action_max)
        self._state_info = (state_size, state_min, state_max)

        self._sock = []
        self._init_struct = init_struct
        self._send_struct = send_struct
        self._recv_struct = recv_struct
        self._init_buf = []
        self._send_buf = []
        self._recv_buf = []

        self._all_observations = []

        super(SocketEnv, self).__init__(
            max_step=max_step,
            action_size=action_size[0], action_min=action_min[0], action_max=action_max[0],
            state_size=state_size[0], state_min=state_min[0], state_max=state_max[0]
        )

    def _reset_phase(self) -> np.ndarray:
        if len(self._init_struct) > 0 and len(self._init_buf) == 0:
            for init_struct in self._init_struct:
                self._init_buf.append(ctypes.create_string_buffer(ctypes.sizeof(init_struct)))

        if len(self._send_buf) == 0:
            for i in range(len(self._send_struct)):
                self._send_buf.append(ctypes.create_string_buffer(ctypes.sizeof(self._send_struct[i])))
                self._recv_buf.append(ctypes.create_string_buffer(ctypes.sizeof(self._recv_struct[i])))

        for sock in self._sock:
            sock.close()
        self._sock.clear()

        for i in range(len(self._ip)):
            self._sock.append(connect(self._ip[i], self._port[i]))

        self._init_phase()

        init_to_send, init_all_obs = self._get_init_content()
        for i in range(len(self._init_buf)):
            struct_pack_into(self._init_struct[i], init_to_send[i], self._init_buf[i])
            self._sock[i].sendall(self._init_buf[i])

        self._all_observations = self._process_observations(init_all_obs)

        return self._get_state(self._all_observations[0])

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        all_actions = self._gen_actions(self._all_observations[1:])
        all_actions.insert(0, action)
        all_obs = []
        for i in range(len(self._send_struct)):
            struct_pack_into(self._send_struct[i], all_actions[i], self._send_buf[i])
            self._sock[i].sendall(self._send_buf[i])

        for i in range(len(self._recv_struct)):
            recv_exact_n_bytes_into(self._sock[i], ctypes.sizeof(self._recv_buf[i]), memoryview(self._recv_buf[i]))
            obs = struct_unpack_from(self._recv_struct[i], self._recv_buf[i])
            all_obs.append(obs)

        self._all_observations = self._process_observations(all_obs)

        return np.array(self._all_observations[0])

    @abc.abstractmethod
    def _get_state(self, obs: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _get_reward(self, obs: np.ndarray, prev_obs: Optional[np.ndarray]) -> Tuple[float, bool]:
        pass

    @abc.abstractmethod
    def _init_phase(self):
        pass

    @abc.abstractmethod
    def _get_init_content(self) -> Tuple[List[List], List[List]]:
        pass

    @abc.abstractmethod
    def _process_observations(self, obs: List[List]) -> List[np.ndarray]:
        pass

    @abc.abstractmethod
    def _gen_actions(self, all_observations: List[np.ndarray]) -> List[np.ndarray]:
        pass
