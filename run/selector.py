import dogfight
import tracking

from trainer import dqn_trainer
from env import combat_env, hierarchy_env
from utils.define import PI, COMBAT_OBS_INFO
from policy.random_policy import RandomPolicy
from policy.greedy_policy import GreedyPolicy
from env.hierarchy_env import HierarchyWrapper
from env.discrete_env import DirectionActionWrapper
from policy.greedy_policy import get_target_euler_diff

import random
import tf_agents
import functools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from absl import app
from absl import logging
from tf_agents.policies import policy_loader


_IP = '127.0.0.1'
_PORT = 10089


def get_state(obs):
    rx, ry, rz = obs[14] - obs[0], obs[15] - obs[1], obs[16] - obs[2]
    roll, pitch, yaw, roll_opp, pitch_opp, yaw_opp = obs[3], obs[4], obs[5], obs[17], obs[18], obs[19]
    state = [rx, ry, rz] + \
            list(get_target_euler_diff(rx, ry, rz, roll, pitch, yaw)[1:]) + \
            list(get_target_euler_diff(-rx, -ry, -rz, roll_opp, pitch_opp, yaw_opp)[1:])

    return np.array(state)


state_min = [-2e5] * 2 + [-1e5] + [-PI, -PI] * 2
state_max = [2e5] * 2 + [1e5] + [PI, PI] * 2


def get_reward(obs, prev_obs):
    damage_cause, damage_suffer, health, opp_health = obs[26], obs[12], obs[13], obs[27]

    if health <= 0. or opp_health <= 0.:
        return -1000. if health < opp_health else 1000., True

    return -0.2, False


def _gen_random_init_pos():
    return [([random.uniform(-2e4, 2e4) for _ in range(2)] + [random.uniform(5e2, 1e4)] +
             [random.uniform(COMBAT_OBS_INFO[1][i], COMBAT_OBS_INFO[2][i]) for i in range(3, 6)]) for _ in range(2)]


def env_constructor():
    return combat_env.CombatEnv(
        ip=_IP, port=_PORT,
        mock_policy_fcn=GreedyPolicy,
        state_size=len(state_min), state_min=state_min, state_max=state_max,
        get_state_fcn=get_state, get_reward_fcn=get_reward,
        max_step=2500,
        gen_init_pos_fcn=_gen_random_init_pos,
        introduce_damage=True
    )


def train_selector_dqn(_):
    parallel_num = 1

    dogfight_path = '/mnt/f/tmp/rl/data/dogfight_sac/eval_policy'
    tracking_path = '/mnt/f/tmp/rl/data/tracking_sac/eval_policy'

    dogfight_policy = policy_loader.load(dogfight_path)
    tracking_policy = policy_loader.load(tracking_path)
    process_action_sac = lambda raw: raw                                        # 用于sac训练的底层策略
    process_action_dqn = lambda raw: DirectionActionWrapper._convert_back(raw)  # 用于dqn训练的底层策略

    hierarchy_wrapper = functools.partial(
        hierarchy_env.HierarchyWrapper,
        policies=[(tracking_policy, tracking.get_state, process_action_sac, tracking.state_min, tracking.state_max)] +
                 [(dogfight_policy, dogfight.get_state, process_action_sac, dogfight.state_min, dogfight.state_max)]
    )

    env = env_constructor()

    def eval_printer(trajectory):
        raw_obs = env.convert_state(trajectory.observation[0], True)
        tf.print('动作: {}, 相对位移: {}, 欧拉角: {}, 奖励: {}'.
                 format(trajectory.action[0], raw_obs[0:3], raw_obs[3:7], trajectory.reward))

    trainer = dqn_trainer.DQNTrainer(
        env_constructor=env_constructor,
        env_wrappers=[hierarchy_wrapper],
        eval_observer=[eval_printer],
        collect_episodes_per_iter=parallel_num,
        train_rounds_per_iter=parallel_num * 12,
        initial_collect_episodes=parallel_num * 3,
        metric_buffer_num=parallel_num,
        q_net_lr=3e-4,
        gamma=0.999,
        reward_scale_factor=1.0,
        replay_cap=1000000,
        train_summary_dir='./save/selector/dqn/summary/train',
        eval_summary_dir='./save/selector/dqn/summary/eval',
        train_checkpoint_dir='./save/selector/dqn/checkpoint/train',
        policy_checkpoint_dir='./save/selector/dqn/checkpoint/policy',
        replay_checkpoint_dir='./save/selector/dqn/checkpoint/replay',
        eval_policy_save_dir='./save/selector/dqn/eval_policy',
        using_reverb=False,
        using_per=False,
        using_ddqn=False,
        parallel_num=parallel_num
    )

    trainer.train()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    tf_agents.system.system_multiprocessing.handle_main(functools.partial(app.run, train_selector_dqn))
