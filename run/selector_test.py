import example

from utils.define import PI
from trainer import dqn_trainer
from env import combat_env, hierarchy_env
from policy.random_policy import RandomPolicy
from policy.greedy_policy import GreedyPolicy
from policy.policy_loader import PolicyLoader
from env.discrete_env import DirectionActionWrapper

import tf_agents
import functools
import numpy as np

from absl import app
from absl import logging
from typing import Tuple
from tf_agents.policies import policy_loader


_IP = '127.0.0.1'
_PORT = 10089


def get_state(obs: np.ndarray) -> np.ndarray:
    state = obs
    return np.array(state)


state_min = [-1e5, -1e5, 0., -PI / 4., -PI / 3., 0., -1e3, -1e3, -1e3, -PI / 4, -PI / 4, -PI / 4, 0., 0.] * 2
state_max = [1e5, 1e5, 1e5, PI / 4., PI / 3., PI * 2., 1e3, 1e3, 1e3, PI / 4, PI / 4, PI / 4, 5., 100.] * 2


def get_reward(obs: np.ndarray, prev_obs: np.ndarray) -> Tuple[float, bool]:
    damage_cause, damage_suffer, health, opp_health = obs[26], obs[12], obs[13], obs[27]

    if health <= 0. or opp_health <= 0.:
        return -500. if health < opp_health else 500., True

    if prev_obs is None:
        return 0., False
    else:
        return (damage_cause - damage_suffer) * 3., False


def env_constructor():
    return combat_env.CombatEnv(
        ip=_IP, port=_PORT,
        mock_policy_fcn=PolicyLoader('./save/example/dqn/eval_policy'),
        mock_policy_info=[example.get_state, DirectionActionWrapper._convert_back, example.state_min, example.state_max],
        # mock_policy_fcn=RandomPolicy,
        state_size=len(state_min), state_min=state_min, state_max=state_max,
        get_state_fcn=get_state, get_reward_fcn=get_reward,
        max_step=2000,
        introduce_damage=True
    )


def get_dqn_trainer():
    parallel_num = 1    # 只能是1

    sac_policy_path = './save/example/sac/eval_policy'
    dqn_policy_path = './save/example/dqn/eval_policy'

    sac_policy = policy_loader.load(sac_policy_path)
    dqn_policy = policy_loader.load(dqn_policy_path)
    process_action_sac = lambda raw: raw                                        # 用于sac训练的底层策略
    process_action_dqn = lambda raw: DirectionActionWrapper._convert_back(raw)  # 用于dqn训练的底层策略

    hierarchy_wrapper = functools.partial(
        hierarchy_env.HierarchyWrapper,
        policies=[(sac_policy, example.get_state, process_action_sac, example.state_min, example.state_max)] +
                 [(dqn_policy, example.get_state, process_action_dqn, example.state_min, example.state_max)]
    )

    return dqn_trainer.DQNTrainer(
        env_constructor=env_constructor,                                    # 无需变动
        env_wrappers=[hierarchy_wrapper],                                   # 无需变动
        collect_episodes_per_iter=parallel_num,                             # 无需变动
        train_rounds_per_iter=parallel_num * 8,                             # 无需变动
        initial_collect_episodes=parallel_num * 3,                          # 无需变动
        metric_buffer_num=parallel_num,                                     # 无需变动
        fc_layer_params=(256, 256),                                         # Critic全连接隐层节点数量（两层各256节点）
        epsilon_greedy=0.1,                                                 # 探索时采用随机策略的概率
        q_net_lr=3e-4,                                                      # Critic网络学习率
        gamma=0.99,                                                         # 折扣因子
        reward_scale_factor=1.0,                                            # 奖励放缩，调整参数使其乘上一局游戏总奖励绝对值小于1000
        target_update_tau=0.005,                                            # 目标网络软更新参数
        target_update_period=1,                                             # 更新目标网络间隔时间
        replay_cap=1000000,                                                 # 经验回放的大小
        train_summary_dir='./save/selector/dqn/summary/train',              # 训练数据记录路径，用于tensorboard可视化
        eval_summary_dir='./save/selector/dqn/summary/eval',                # 验证数据记录路径，用于tensorboard可视化
        train_checkpoint_dir='./save/selector/dqn/checkpoint/train',        # 用于中断后继续训练
        policy_checkpoint_dir='./save/selector/dqn/checkpoint/policy',      # 用于中断后继续训练
        replay_checkpoint_dir='./save/selector/dqn/checkpoint/replay',      # 用于中断后继续训练
        eval_policy_save_dir='./save/selector/dqn/eval_policy',             # 训练好的策略保存的位置
        using_ddqn=False,                                                   # 是否使用Double-DQN算法
        parallel_num=parallel_num                                           # 并行采样进程数
    )


def main(_):
    trainer = get_dqn_trainer()
    trainer.train()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    tf_agents.system.system_multiprocessing.handle_main(functools.partial(app.run, main))
