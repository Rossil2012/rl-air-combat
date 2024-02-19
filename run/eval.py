import dogfight
import tracking

from env import combat_env
from utils.define import PI, COMBAT_OBS_INFO
from env.base_env import norm, denorm
from policy.pid_policy import get_target_euler_diff

import math
import random
import collections
import numpy as np
import matplotlib.pyplot as plt

from tf_agents.policies import policy_loader
from policy.random_policy import RandomPolicy
from env.hierarchy_env import HierarchyWrapper


_IP = '127.0.0.1'
_PORT = 10089

def get_state(obs):
    rx, ry, rz = obs[14] - obs[0], obs[15] - obs[1], obs[16] - obs[2]
    roll, pitch, yaw, roll_opp, pitch_opp, yaw_opp = obs[3], obs[4], obs[5], obs[17], obs[18], obs[19]
    state = [rx, ry, rz] + \
            list(get_target_euler_diff(rx, ry, rz, roll, pitch, yaw)[1:]) + \
            list(get_target_euler_diff(-rx, -ry, -rz, roll_opp, pitch_opp, yaw_opp)[1:])

    return np.array(state)


def get_reward(obs, prev_obs):
    damage_cause, damage_suffer, health, opp_health = obs[26], obs[12], obs[13], obs[27]

    if health <= 0. or opp_health <= 0.:
        return -1000. if health < opp_health else 1000., True

    return -0.2, False


state_min = [-2e5] * 2 + [-1e5] + [-PI, -PI] * 2
state_max = [2e5] * 2 + [1e5] + [PI, PI] * 2


def _gen_random_init_pos():
    return [([random.uniform(-2e4, 2e4) for _ in range(2)] + [random.uniform(5e2, 1e4)] +
             [random.uniform(COMBAT_OBS_INFO[1][i], COMBAT_OBS_INFO[2][i]) for i in range(3, 6)]) for _ in range(2)]


def plot_dist_choice(dist_choice_all):
    plt.figure()
    ordered = sorted(dist_choice_all, key=lambda s: s[0])
    tot = [0, 0]
    for _, choice in ordered:
        tot[choice] += 1

    cur = [0, 0]
    plot = [[], []], [[], []]
    for dist, choice in ordered:
        cur[choice] += 1
        if dist < 300:
            continue
        for i in range(2):
            plot[i][0].append(dist)
            plot[i][1].append(cur[i] / float(cur[0] + cur[1]))

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    label = ['跟踪子策略', '狗斗子策略']
    color = ['red', 'blue']
    for i in range(2):
        plt.plot(plot[i][0], plot[i][1], color=color[i], label=label[i])

    plt.axvline(3000 * math.sqrt(3) / 2, ls='-.', c='green', lw=1)

    plt.legend(loc="best")
    plt.xlabel('距离', fontsize=14)
    plt.ylabel('比例', fontsize=14)

    plt.show()


def plot_step_and_reward(step_list, reward_list):
    x_list = list(range(len(step_list)))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.figure()
    plt.plot(x_list, step_list, color='blue', label='步长')
    plt.legend(loc="best")
    plt.xlabel('轮次', fontsize=15)
    plt.ylabel('步长', fontsize=15)
    plt.show()

    plt.figure()
    plt.plot(x_list, reward_list, color='red', label='奖励')
    plt.legend(loc="best")
    plt.xlabel('轮次', fontsize=15)
    plt.ylabel('奖励', fontsize=15)
    plt.show()


def plot():
    # dogfight_path = '/mnt/f/tmp/rl/data/dogfight_sac/eval_policy'
    # tracking_path = '/mnt/f/tmp/rl/data/tracking_sac/eval_policy'
    # selector_path = '/mnt/f/tmp/rl/data/selector_dqn/eval_policy'
    dogfight_path = 'F:/tmp/rl/data/dogfight_sac/eval_policy'
    tracking_path = 'F:/tmp/rl/data/tracking_sac/eval_policy'
    selector_path = 'F:/tmp/rl/data/selector_dqn/eval_policy'

    dogfight_policy = policy_loader.load(dogfight_path)
    tracking_policy = policy_loader.load(tracking_path)
    selector_policy = policy_loader.load(selector_path)

    env = combat_env.CombatEnv(
        ip=_IP, port=_PORT,
        mock_policy_fcn=RandomPolicy,
        state_size=len(state_min), state_min=state_min, state_max=state_max,
        get_state_fcn=get_state, get_reward_fcn=get_reward,
        max_step=10000,
        gen_init_pos_fcn=_gen_random_init_pos,
        introduce_damage=True
    )

    process_action_sac = lambda raw: raw
    h_env = HierarchyWrapper(
        env=env,
        policies=[(tracking_policy, tracking.get_state, process_action_sac, tracking.state_min, tracking.state_max)] +
                 [(dogfight_policy, dogfight.get_state, process_action_sac, dogfight.state_min, dogfight.state_max)]
    )

    def iter_once():
        ts = h_env.reset()
        tot_reward = 0.
        cur_step = 0
        dist_choice = []
        while not ts.is_last():
            cur_step += 1
            choice = selector_policy.action(ts).action
            ts = h_env.step(choice)
            tot_reward += ts.reward

            rx, ry, rz = denorm(np.array(ts.observation), state_min, state_max)[0:3]
            dist = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

            dist_choice.append((dist, choice))
        return cur_step, tot_reward, dist_choice

    dist_choice_all, step_list, reward_list = [], [], []
    for i in range(5000):
        max_step, reward, dist_choice = iter_once()
        dist_choice_all.extend(dist_choice)
        step_list.append(max_step)
        reward_list.append(reward)
        print(i, max_step, reward)

    plot_step_and_reward(step_list, reward_list)
    plot_dist_choice(dist_choice_all)


if __name__ == '__main__':
    plot()
