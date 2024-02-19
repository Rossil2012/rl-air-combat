from env import combat_env
from policy.pid_policy import PIDPolicy
from utils.define import COMBAT_OBS_INFO, PI

import random
import numpy as np
from tf_agents.policies.random_py_policy import RandomPyPolicy

_IP = '127.0.0.1'
_PORT = 10089


def get_state(obs: np.ndarray) -> np.ndarray:
    state = obs
    return np.array(state)


state_size = COMBAT_OBS_INFO[0]
state_min = COMBAT_OBS_INFO[1]
state_max = COMBAT_OBS_INFO[2]


def get_reward(obs, prev_obs):
    damage_cause, damage_suffer, health, opp_health = obs[26], obs[12], obs[13], obs[27]

    # 如果任意一方生命值降为0，给予高奖励或惩罚，并终止当前episode
    if health <= 0. or opp_health <= 0.:
        return -500. if health < opp_health else 500., True

    # 需要特殊判断prev_obs是否为None
    if prev_obs is None:
        return 0., False
    else:
        return (damage_cause - damage_suffer) * 3., False


def _gen_random_init_pos():
    return [([random.uniform(-2e4, 2e4) for _ in range(2)] + [random.uniform(5e2, 1e4)] +
             [random.uniform(COMBAT_OBS_INFO[1][i], COMBAT_OBS_INFO[2][i]) for i in range(3, 6)]) for _ in range(2)]


def env_constructor():
    return combat_env.CombatEnv(
        ip=_IP, port=_PORT,
        mock_policy_fcn=PIDPolicy,
        state_size=state_size, state_min=state_min, state_max=state_max,
        get_state_fcn=get_state, get_reward_fcn=get_reward,
        gen_init_pos_fcn=_gen_random_init_pos,
        max_step=2000,
        introduce_damage=True
    )


def pid():
    py_env = env_constructor()
    policy = PIDPolicy()
    time_step = py_env.reset()
    tot_reward = 0.
    while not time_step.is_last():
        policy_step = policy.action(time_step)
        time_step = py_env.step(policy_step.action)
        tot_reward += time_step.reward

    print(tot_reward)


if __name__ == '__main__':
    pid()