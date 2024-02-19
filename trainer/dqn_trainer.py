from env.base_env import BaseEnv
from trainer.local_trainer import LocalTrainer

import functools
import tensorflow as tf

from types import SimpleNamespace
from typing import List, Callable
from tf_agents.utils import common
from tf_agents.utils import lazy_loader
from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential
from tf_agents.train.utils import spec_utils
from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.replay_buffers.reverb_replay_buffer import ReverbReplayBuffer
from tf_agents.replay_buffers.reverb_utils import ReverbAddTrajectoryObserver
reverb = lazy_loader.LazyLoader('reverb', globals(), 'reverb')


logits = functools.partial(
    tf.keras.layers.Dense,
    activation=None,
    kernel_initializer=tf.random_uniform_initializer(minval=-0.03, maxval=0.03),
    bias_initializer=tf.constant_initializer(-0.2))


dense = functools.partial(
    tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.compat.v1.variance_scaling_initializer(
        scale=2.0, mode='fan_in', distribution='truncated_normal'))


def create_feedforward_network(fc_layer_units, num_actions):
    return sequential.Sequential([dense(num_units) for num_units in fc_layer_units] + [logits(num_actions)])


class DQNTrainer(LocalTrainer):
    def __init__(self,
                 env_constructor: Callable[[], BaseEnv],
                 env_wrappers: List[Callable] = None,
                 eval_observer: List[Callable] = None,
                 fc_layer_params=(256, 256),
                 epsilon_greedy=0.1,
                 n_step_update=1,
                 q_net_lr=3e-4,
                 gamma=0.99,
                 reward_scale_factor=1.0,
                 target_update_tau=0.005,
                 target_update_period=1,
                 batch_size=256,
                 replay_cap=16 * 1000,
                 initial_collect_episodes: int = 3,
                 metric_buffer_num: int = 10,
                 summaries_flush_secs: int = 5,
                 train_summary_dir: str = './save/dqn/train',
                 eval_summary_dir: str = './save/dqn/eval',
                 train_checkpoint_dir: str = './save/dqn/checkpoint/train',
                 policy_checkpoint_dir: str = './save/dqn/checkpoint/policy',
                 replay_checkpoint_dir: str = './save/dqn/checkpoint/replay',
                 eval_policy_save_dir: str = './save/dqn/eval_policy',
                 collect_episodes_per_iter: int = 1,
                 train_rounds_per_iter: int = 4,
                 eval_interval: int = 10,
                 eval_episodes: int = 1,
                 max_iterations: int = 100000,
                 using_reverb: bool = False,
                 using_per: bool = False,
                 using_ddqn: bool = False,
                 parallel_num: int = 1
                 ):

        env = self._construct_tfa_env(env_constructor, env_wrappers or [], 1)[0]
        observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(env)

        num_actions = action_spec.maximum - action_spec.minimum + 1
        q_net = create_feedforward_network(fc_layer_params, num_actions)

        train_step = tf.compat.v1.train.get_or_create_global_step()
        make_agent_fcn = DdqnAgent if using_ddqn else DqnAgent
        agent = make_agent_fcn(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=q_net,
            epsilon_greedy=epsilon_greedy,
            n_step_update=n_step_update,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=q_net_lr),
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=None,
            debug_summaries=False,
            summarize_grads_and_vars=False,
            train_step_counter=train_step
        )

        replay_buffer = None
        replay_observer = None
        replay_checkpointer = None

        if using_reverb:
            table_name = 'DQN_PER'
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
            eval_policy = PyTFEagerPolicy(agent.policy, use_tf_function=True)
        else:
            using_py = False
            initial_collect_policy = RandomTFPolicy(time_step_spec, action_spec)
            collect_policy = agent.collect_policy
            eval_policy = agent.policy

        super(DQNTrainer, self).__init__(
            env_constructor=env_constructor,
            global_step=train_step,
            agent=agent,
            initial_collect_policy=initial_collect_policy,
            collect_policy=collect_policy,
            eval_policy=eval_policy,
            eval_observer=eval_observer,
            env_wrappers=env_wrappers,
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
