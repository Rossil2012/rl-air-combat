import abc


class RLTrainer(metaclass=abc.ABCMeta):
    def __init__(self, max_iterations: int = 100000):
        self._max_iterations = max_iterations

    def train(self):
        for step in range(1, self._max_iterations + 1):
            if self._iter_one_step(step):
                break

    @abc.abstractmethod
    def _iter_one_step(self, cur_step):
        """
        单步训练操作
        :return: 返回True，episode终止；否则返回False。
        """
