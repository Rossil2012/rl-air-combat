import selector
import tracking
import dogfight

from env.display_env import DisplayEnv
from policy.lazy_policy import LazyPolicy
from policy.random_policy import RandomPolicy
from policy.greedy_policy import GreedyPolicy
from env.discrete_env import DirectionActionWrapper
from utils.define import ACTION_INFO, COMBAT_OBS_INFO

import time
import random
import tensorflow as tf

from tf_agents.policies import policy_loader

_IP = ['127.0.0.1'] * 2
_PORT = [20000, 20001]


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


def _gen_random_init_pos():
    return [random.uniform(-3e3, 3e3) for _ in range(2)] + [random.uniform(3e3, 4e3)] + \
           [random.uniform(COMBAT_OBS_INFO[1][i], COMBAT_OBS_INFO[2][i]) for i in range(3, 6)]


def _hierarchical_policy_display_env():
    opp_policy, opp_state_fcn, opp_state_info, opp_action_wrapper = _make_random_policy()
    # opp_policy, opp_state_fcn, opp_state_info, opp_action_wrapper = _make_lazy_policy()
    # opp_policy, opp_state_fcn, opp_state_info, opp_action_wrapper = _make_greedy_policy()

    policy_path = r'F:\tmp\rl\data\selector_dqn\eval_policy'
    policy, state_fcn, state_info, _ = _make_policy_loader(selector, policy_path, using_dqn=True)

    tracking_sac_path = r'F:\tmp\rl\data\tracking_sac\eval_policy'
    tracking_sac_info = _make_policy_loader(tracking, tracking_sac_path, using_dqn=False)

    dogfight_sac_path = r'F:\tmp\rl\data\dogfight_sac\eval_policy'
    dogfight_sac_info = _make_policy_loader(dogfight, dogfight_sac_path, using_dqn=False)

    display = DisplayEnv(
        ip=_IP, port=_PORT,

        # 修改第一项，第二项是靶机
        state_size=[state_info[0], opp_state_info[0]],
        state_min=[state_info[1], opp_state_info[1]],
        state_max=[state_info[2], opp_state_info[2]],
        get_state_fcn=[state_fcn, opp_state_fcn],
        policies=[policy, opp_policy],
        policy_info=[[tracking_sac_info, dogfight_sac_info], None],  # 顺序要和分层策略中子策略顺序一致
        action_wrappers=[None, opp_action_wrapper],
        gen_init_pos_fcn=_gen_random_init_pos,
        max_step=3000
    )

    return display


def _one_policy_display_env():
    # opp_policy, opp_state_fcn, opp_state_info, opp_action_wrapper = _make_random_policy()
    opp_policy, opp_state_fcn, opp_state_info, opp_action_wrapper = _make_lazy_policy()
    # opp_policy, opp_state_fcn, opp_state_info, opp_action_wrapper = _make_greedy_policy()

    policy_path = r'F:\tmp\rl\save\dogfight\sac_140k\eval_policy'
    policy, state_fcn, state_info, action_wrapper = _make_policy_loader(selector, policy_path, using_dqn=False)
    # policy, state_fcn, state_info, action_wrapper = _make_greedy_policy()


    display = DisplayEnv(
        ip=_IP, port=_PORT,

        # 修改第一项，第二项是靶机
        state_size=[state_info[0], opp_state_info[0]],
        state_min=[state_info[1], opp_state_info[1]],
        state_max=[state_info[2], opp_state_info[2]],
        get_state_fcn=[state_fcn, opp_state_fcn],
        policies=[policy, opp_policy],
        action_wrappers=[action_wrapper, opp_action_wrapper],
        gen_init_pos_fcn=_gen_random_init_pos,
        max_step=3000
    )

    return display


def main():
    display = _one_policy_display_env()
    # display = _hierarchical_policy_display_env()

    ending = False
    for _ in range(1):  # 连续查看五局游戏
        display.reset()
        while not ending:
            time.sleep(0.06)  # 放慢速度
            ending = display.step()
        ending = False
        time.sleep(5.)  # 等待训练环境重启


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # Tensorflow启动时默认占用所有显存，这个设置可以减少显存占用，防止训练环境卡顿
    with tf.compat.v1.Session(config=config).as_default():
        main()
