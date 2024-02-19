from env import discrete_env, combat_env
from utils.define import PI, COMBAT_OBS_INFO
from trainer import dqn_trainer, sac_trainer
from policy.random_policy import RandomPolicy
from policy.greedy_policy import GreedyPolicy
from policy.greedy_policy import get_target_euler_diff

import math
import random
import tf_agents
import functools
import numpy as np
import tensorflow as tf

from absl import app
from absl import logging
from tf_agents.policies.random_py_policy import RandomPyPolicy


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
    rx, ry, rz, pitch_diff, yaw_diff, pitch_diff_opp, yaw_diff_opp = get_state(obs)
    dist = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    if health <= 0. or opp_health <= 0.:
        return -800. if health < opp_health else 800., True

    def angle_reward(now, prev, factor):
        return (abs(prev[0]) - abs(now[0])) * factor + (abs(prev[1]) - abs(now[1])) * factor

    def dist_reward(now, prev, factor):
        return (prev - now) * factor - (30. if now < 300. else 0.)

    def damage_reward(cause, suffer):
        return cause * 5 - suffer * 3

    if prev_obs is not None:
        rx_p, ry_p, rz_p, pitch_diff_p, yaw_diff_p, pitch_diff_opp_p, yaw_diff_opp_p = get_state(prev_obs)
        prev_dist = math.sqrt(rx_p ** 2 + ry_p ** 2 + rz_p ** 2)
        return -1. + dist_reward(dist, prev_dist, 0.1) + damage_reward(damage_cause, damage_suffer) + \
               angle_reward([pitch_diff, yaw_diff], [pitch_diff_p, yaw_diff_p], 40) - \
               angle_reward([pitch_diff_opp, yaw_diff_opp], [pitch_diff_opp_p, yaw_diff_opp_p], 40), False
    else:
        return -1., False


def _gen_init_pos():
    obs_min = COMBAT_OBS_INFO[1][0:6]
    obs_max = COMBAT_OBS_INFO[2][0:6]
    pos_a = [random.uniform(obs_min[i], obs_max[i]) for i in range(6)]
    pos_b = [(pos_a[i] + random.uniform(-3000., 3000.)) for i in range(0, 3)] + \
            [random.uniform(obs_min[i], obs_max[i]) for i in range(3, 6)]

    pos_a[2] = max(pos_a[2], 200.)
    pos_b[2] = max(pos_b[2], 200.)

    return [pos_a, pos_b]


def env_constructor():
    return combat_env.CombatEnv(
        ip=_IP, port=_PORT,
        mock_policy_fcn=GreedyPolicy,
        state_size=len(state_min), state_min=state_min, state_max=state_max,
        get_state_fcn=get_state, get_reward_fcn=get_reward, gen_init_pos_fcn=_gen_init_pos,
        max_step=800,
        introduce_damage=True
    )


def get_sac_trainer():
    parallel_num = 24

    env = env_constructor()

    def eval_printer(trajectory):
        raw_state = env.convert_state(trajectory.observation[0], True)
        raw_action = env.convert_action(trajectory.action[0], True)
        tf.print('动作: {}, 相对位移: {}, 相对速度: {}, 欧拉角: {}, 奖励: {}'.
                 format(raw_action, raw_state[0:3], raw_state[3:6], raw_state[6:10], trajectory.reward))

    return sac_trainer.SACTrainer(
        env_constructor=env_constructor,
        eval_observer=[eval_printer],
        collect_episodes_per_iter=parallel_num,
        train_rounds_per_iter=parallel_num * 10,
        initial_collect_episodes=parallel_num * 3,
        metric_buffer_num=parallel_num,
        train_summary_dir='./save/dogfight/sac/summary/train',
        eval_summary_dir='./save/dogfight/sac/summary/eval',
        train_checkpoint_dir='./save/dogfight/sac/checkpoint/train',
        policy_checkpoint_dir='./save/dogfight/sac/checkpoint/policy',
        replay_checkpoint_dir='./save/dogfight/sac/checkpoint/replay',
        eval_policy_save_dir='./save/dogfight/sac/eval_policy',
        critic_lr=3e-4,
        actor_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.997,
        reward_scale_factor=1.0,
        target_update_tau=0.003,
        target_update_period=20,
        replay_cap=1000000,
        using_reverb=False,
        using_per=False,
        parallel_num=parallel_num
    )


def get_dqn_trainer():
    parallel_num = 24

    env = env_constructor()

    def eval_printer(trajectory):
        raw_obs = env.convert_state(trajectory.observation[0], True)
        tf.print('动作: {}, 相对位移: {}, 欧拉角: {}, 奖励: {}'.
                 format(trajectory.action[0], raw_obs[0:3], raw_obs[3:7], trajectory.reward))

    return dqn_trainer.DQNTrainer(
        env_constructor=env_constructor,
        eval_observer=[eval_printer],
        env_wrappers=[discrete_env.DirectionActionWrapper],
        collect_episodes_per_iter=parallel_num,
        train_rounds_per_iter=parallel_num * 6,
        initial_collect_episodes=parallel_num * 3,
        metric_buffer_num=parallel_num,
        q_net_lr=3e-4,
        gamma=0.997,
        target_update_tau=0.01,
        target_update_period=10,
        reward_scale_factor=1.0,
        replay_cap=1000000,
        train_summary_dir='./save/dogfight/dqn/summary/train',
        eval_summary_dir='./save/dogfight/dqn/summary/eval',
        train_checkpoint_dir='./save/dogfight/dqn/checkpoint/train',
        policy_checkpoint_dir='./save/dogfight/dqn/checkpoint/policy',
        replay_checkpoint_dir='./save/dogfight/dqn/checkpoint/replay',
        eval_policy_save_dir='./save/dogfight/dqn/eval_policy',
        using_reverb=False,
        using_per=False,
        using_ddqn=True,
        parallel_num=parallel_num
    )


def train(_):
    trainer = get_sac_trainer()
    trainer.train()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    tf_agents.system.system_multiprocessing.handle_main(functools.partial(app.run, train))
