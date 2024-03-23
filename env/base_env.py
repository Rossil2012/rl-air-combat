import abc
import numpy as np

from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step
from typing import Iterable, Tuple, Optional
from tf_agents.environments import PyEnvironment


def norm(before, min_list, max_list):
    after = np.clip(before, min_list, max_list)
    after = ((after - min_list) / (np.array(max_list) - min_list) - 0.5) * 2
    return after


def denorm(after, min_list, max_list):
    before = (after / 2 + 0.5) * (np.array(max_list) - min_list) + min_list
    return before


class BaseEnv(PyEnvironment):
    def __init__(self,
                 action_size: int, action_min: Iterable, action_max: Iterable,
                 state_size: int, state_min: Iterable, state_max: Iterable,
                 max_step: int):
        super(BaseEnv, self).__init__(handle_auto_reset=True)

        self._action_spec = BoundedArraySpec((action_size,), np.float32,
                                             minimum=-1, maximum=1, name='action')
        self._state_spec = BoundedArraySpec((state_size,), np.float32,
                                            minimum=-1, maximum=1, name='observation')

        self._action_min, self._action_max = action_min, action_max
        self._state_min, self._state_max = state_min, state_max

        self._cur_step = 0
        self._max_step = max_step
        self._prev_obs = None

    def convert_action(self, action, is_norm: bool):
        fcn = denorm if is_norm else norm
        return fcn(action, self._action_min, self._action_max)

    def convert_state(self, state, is_norm: bool):
        fcn = denorm if is_norm else norm
        return fcn(state, self._state_min, self._state_max)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._state_spec

    def _reset(self):
        raw_init_state = self._reset_phase()
        norm_init_state = norm(raw_init_state, self._state_min, self._state_max)
        self._cur_step = 0
        self._prev_obs = None
        return time_step.restart(norm_init_state.astype(np.float32))

    def _step(self, norm_action):
        self._cur_step += 1

        raw_action = denorm(norm_action, self._action_min, self._action_max)
        obs = self._apply_action(raw_action)

        raw_state = self._get_state(obs)
        norm_state = norm(raw_state, self._state_min, self._state_max)

        reward, done = self._get_reward(obs, self._prev_obs)
        self._prev_obs = obs

        if self._cur_step == self._max_step or done:
            self._cur_step = 0
            return time_step.termination(observation=norm_state.astype(np.float32), reward=reward)
        else:
            return time_step.transition(observation=norm_state.astype(np.float32), reward=reward)

    @abc.abstractmethod
    def _reset_phase(self) -> np.ndarray:
        """
        TODO:
        :return:
        """

    @abc.abstractmethod
    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """
        TODO
        :return:
        """

    @abc.abstractmethod
    def _get_state(self, obs: np.ndarray) -> np.ndarray:
        """
        TODO
        :return:
        """

    @abc.abstractmethod
    def _get_reward(self, obs: np.ndarray, prev_obs: Optional[np.ndarray]) -> Tuple[float, bool]:
        """
        TODO
        :return:
        """