import dogfight
from env.online_env import OnlineEnv
from policy.lazy_policy import LazyPolicy
from policy.random_policy import RandomPolicy
from policy.greedy_policy import GreedyPolicy
from env.discrete_env import DirectionActionWrapper
from utils.define import ACTION_INFO, COMBAT_OBS_INFO

import time
import random
import tensorflow as tf

from tf_agents.policies import policy_loader


_IP = '127.0.0.1'
_PORT = 10000


def _make_policy_loader(pkg, path, using_dqn):
    policy = policy_loader.load(path)
    get_state_fcn = pkg.get_state
    state_info = len(pkg.state_min), pkg.state_min, pkg.state_max
    action_wrapper = DirectionActionWrapper._convert_back if using_dqn else None
    return policy, get_state_fcn, state_info, action_wrapper


def _make_random_policy():
    random_policy = RandomPolicy()
    get_state_fcn = lambda _: _
    return random_policy, get_state_fcn, COMBAT_OBS_INFO, None


def _make_lazy_policy():
    lazy_policy = LazyPolicy()
    get_state_fcn = lambda _: _
    return lazy_policy, get_state_fcn, COMBAT_OBS_INFO, None


def _make_greedy_policy():
    greedy_policy = GreedyPolicy()
    get_state_fcn = lambda _: _
    return greedy_policy, get_state_fcn, COMBAT_OBS_INFO, None


def _one_policy_online_env():
    policy_path = r'F:\tmp\rl\save\dogfight\sac_140k\eval_policy'
    policy, state_fcn, state_info, action_wrapper = _make_policy_loader(dogfight, policy_path, using_dqn=False)
    # policy, state_fcn, state_info, action_wrapper = _make_greedy_policy()

    online = OnlineEnv(
        ip=_IP, port=_PORT,
        state_size=state_info[0], state_min=state_info[1], state_max=state_info[2],
        get_state_fcn=state_fcn,
        policy=policy,
        action_wrapper=action_wrapper
    )

    return online


def main():
    online = _one_policy_online_env()

    ending = False
    online.reset()

    while not ending:
        ending = online.step()


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # Tensorflow启动时默认占用所有显存，这个设置可以减少显存占用，防止训练环境卡顿
    with tf.compat.v1.Session(config=config).as_default():
        main()
