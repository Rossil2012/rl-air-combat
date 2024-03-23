import numpy as np

from tf_agents.typing.types import NestedArray
from tf_agents.environments import PyEnvironment
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper


_SPEED_LEVEL = 5
_DIRECTION_NUM = 5


class DirectionActionWrapper(PyEnvironmentBaseWrapper):
    def __init__(self, env: PyEnvironment):
        super(DirectionActionWrapper, self).__init__(env)
        self._action_spec = BoundedArraySpec((), np.int32, minimum=0, maximum=5 * _SPEED_LEVEL - 1)

    def _step(self, action):
        origin_action = self._convert_back(action)
        return self._env.step(origin_action)

    def action_spec(self) -> NestedArray:
        return self._action_spec

    @classmethod
    def _convert_back(cls, after):
        """
        after % _DIRECTION_NUM == 0: 上
        after % _DIRECTION_NUM == 1: 下
        after % _DIRECTION_NUM == 2: 左
        after % _DIRECTION_NUM == 3: 右
        after % _DIRECTION_NUM == 4: 直行
        """
        def action_map(ad):
            direction, speed = ad % _DIRECTION_NUM, (ad // _DIRECTION_NUM) / (_SPEED_LEVEL - 1)
            if direction == 0:
                ac = [0., 1., 0.]
            elif direction == 1:
                ac = [0., -1., 0.]
            elif direction == 2:
                ac = [0., 0., 1.]
            elif direction == 3:
                ac = [0., 0., -1.]
            elif direction == 4:
                ac = [0., 0., 0.]
            else:
                assert 0
            return ac + [speed]

        if after.ndim == 0:
            ret = action_map(after)
        else:
            ret = list(map(action_map, after))

        return np.array(ret, dtype=np.float32)
