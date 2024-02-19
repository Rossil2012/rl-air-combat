from env.base_env import BaseEnv
from trainer.local_trainer import LocalTrainer

import tensorflow as tf

from types import SimpleNamespace
from typing import List, Callable
from tf_agents.utils import lazy_loader
from tf_agents.specs import tensor_spec
from tf_agents.train.utils import spec_utils
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.policies.greedy_policy import GreedyPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.replay_buffers.reverb_replay_buffer import ReverbReplayBuffer
from tf_agents.replay_buffers.reverb_utils import ReverbAddTrajectoryObserver
reverb = lazy_loader.LazyLoader('reverb', globals(), 'reverb')


class SACTrainer(LocalTrainer):
    def __init__(self,
                 env_constructor: Callable[[], BaseEnv],
                 env_wrappers: List[Callable] = None,
                 eval_observer: List[Callable] = None,
                 actor_fc_layer_params=(256, 256),
                 critic_observation_fc_layer_params=(128, 128),
                 critic_action_fc_params=(128, 128),
                 critic_joint_fc_layer_params=(256, 256),
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 alpha_lr=3e-4,
                 gamma=0.99,
                 reward_scale_factor=1.0,
                 target_update_tau=0.005,
                 target_update_period=1,
                 batch_size=256,
                 replay_cap=16 * 1000,
                 initial_collect_episodes: int = 3,
                 metric_buffer_num: int = 10,
                 summaries_flush_secs: int = 5,
                 train_summary_dir: str = './save/sac/train',
                 eval_summary_dir: str = './save/sac/eval',
                 train_checkpoint_dir: str = './save/sac/checkpoint/train',
                 policy_checkpoint_dir: str = './save/sac/checkpoint/policy',
                 replay_checkpoint_dir: str = './save/sac/checkpoint/replay',
                 eval_policy_save_dir: str = './save/sac/eval_policy',
                 collect_episodes_per_iter: int = 1,
                 train_rounds_per_iter: int = 4,
                 eval_interval: int = 10,
                 eval_episodes: int = 1,
                 max_iterations: int = 100000,
                 using_reverb: bool = False,
                 using_per: bool = False,
                 parallel_num: int = 1
                 ):

        env = self._construct_tfa_env(env_constructor, env_wrappers or [], 1)[0]
        observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(env)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork
        )

        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=critic_observation_fc_layer_params,
            action_fc_layer_params=critic_action_fc_params,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform'
        )

        train_step = tf.compat.v1.train.get_or_create_global_step()
        agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(learning_rate=actor_lr),
            critic_optimizer=tf.keras.optimizers.Adam(learning_rate=critic_lr),
            alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=alpha_lr),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            train_step_counter=train_step
        )

        replay_buffer = None
        replay_observer = None
        replay_checkpointer = None

        if using_reverb:
            table_name = 'SAC_PER'
            table = reverb.Table(
                name=table_name,
                max_size=replay_cap,
                sampler=reverb.selectors.Prioritized(0.8) if using_per else reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                rate_limiter=reverb.rate_limiters.MinSize(100),
            )
            reverb_server = reverb.Server(
                tables=[table],
                checkpointer=reverb.checkpointers.DefaultCheckpointer(replay_checkpoint_dir)
            )

            replay_buffer = ReverbReplayBuffer(
                data_spec=tensor_spec.from_spec(agent.collect_data_spec),
                sequence_length=2,
                table_name=table_name,
                local_server=reverb_server
            )

            replay_checkpointer = SimpleNamespace(
                save=lambda *__, **_: replay_buffer.py_client.checkpoint(),
                initialize_or_restore=lambda *__, **_: None
            )

            replay_observer = [
                ReverbAddTrajectoryObserver(
                    py_client=replay_buffer.py_client,
                    table_name=table_name,
                    sequence_length=2,
                    stride_length=1
                )
            ]

            using_py = True
            initial_collect_policy = RandomPyPolicy(time_step_spec, action_spec)
            collect_policy = PyTFEagerPolicy(agent.collect_policy, use_tf_function=True)
            eval_policy = PyTFEagerPolicy(GreedyPolicy(agent.policy), use_tf_function=True)
        else:
            using_py = False
            initial_collect_policy = RandomTFPolicy(time_step_spec, action_spec)
            collect_policy = agent.collect_policy
            eval_policy = GreedyPolicy(agent.policy)

        super(SACTrainer, self).__init__(
            env_constructor=env_constructor,
            env_wrappers=env_wrappers,
            global_step=train_step,
            agent=agent,
            initial_collect_policy=initial_collect_policy,
            collect_policy=collect_policy,
            eval_policy=eval_policy,
            eval_observer=eval_observer,
            replay_buffer=replay_buffer,
            replay_observer=replay_observer,
            replay_checkpointer=replay_checkpointer,
            replay_cap=replay_cap,
            batch_size=batch_size,
            initial_collect_episodes=initial_collect_episodes,
            metric_buffer_num=metric_buffer_num,
            summaries_flush_secs=summaries_flush_secs,
            train_summary_dir=train_summary_dir,
            eval_summary_dir=eval_summary_dir,
            train_checkpoint_dir=train_checkpoint_dir,
            policy_checkpoint_dir=policy_checkpoint_dir,
            replay_checkpoint_dir=replay_checkpoint_dir,
            eval_policy_save_dir=eval_policy_save_dir,
            collect_episodes_per_iter=collect_episodes_per_iter,
            train_rounds_per_iter=train_rounds_per_iter,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            max_iterations=max_iterations,
            using_py=using_py,
            parallel_num=parallel_num
        )
