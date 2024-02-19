from env.base_env import BaseEnv
from trainer.rl_trainer import RLTrainer

import tensorflow as tf

from typing import Union, Callable, List
from tf_agents.train import Actor
from tf_agents.agents import TFAgent
from tf_agents.specs import tensor_spec
from tf_agents.eval import metric_utils
from tf_agents.utils.common import Checkpointer
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.metrics import tf_metrics, py_metrics
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.environments import TFPyEnvironment, ParallelPyEnvironment
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.replay_buffers.py_uniform_replay_buffer import PyUniformReplayBuffer


class _SummaryMetric:
    def __init__(self, global_step, summary_writer):
        self._global_step = global_step
        self._summary_writer = summary_writer
        self._tot_steps = tf.Variable(0, dtype=tf.int64)
        self._episode_length = tf.Variable(0, dtype=tf.int64)
        self._return_sum = tf.Variable(0., dtype=tf.float32)

    def __call__(self, trajectory):
        self._tot_steps.assign_add(1)
        self._episode_length.assign_add(1)
        self._return_sum.assign_add(trajectory.reward)
        with self._summary_writer.as_default():
            tf.summary.scalar(name='Metrics/' + 'Environment Steps', data=self._tot_steps, step=self._global_step)
            if trajectory.is_last():
                tf.summary.scalar(name='Metrics/' + 'Episode Length', data=self._episode_length, step=self._global_step)
                tf.summary.scalar(name='Metrics/' + 'Episode Return', data=self._return_sum, step=self._global_step)
                self._episode_length.assign(0)
                self._return_sum.assign(0.)


def _make_driver(env, policy, num_episodes, observers, train_step, using_py):
    if using_py:
        return Actor(
            env=env,
            policy=policy,
            observers=observers,
            episodes_per_run=num_episodes,
            train_step=train_step
        )
    else:
        return DynamicEpisodeDriver(
            env=env,
            policy=policy,
            observers=observers,
            num_episodes=num_episodes
        )


class LocalTrainer(RLTrainer):
    def __init__(self,
                 env_constructor: Callable[[], BaseEnv],
                 global_step: tf.Variable,
                 agent: TFAgent,
                 collect_policy: Union[TFPolicy, PyPolicy],
                 eval_policy: Union[TFPolicy, PyPolicy],
                 eval_observer: List[Callable] = None,
                 initial_collect_policy: Union[TFPolicy, PyPolicy] = None,
                 env_wrappers: List[Callable] = None,
                 replay_buffer: ReplayBuffer = None,
                 replay_observer: list = None,
                 replay_checkpointer: object = None,
                 replay_cap: int = 32 * 1000,
                 batch_size: int = 256,
                 initial_collect_episodes: int = 3,
                 metric_buffer_num: int = 10,
                 summaries_flush_secs: int = 5,
                 train_summary_dir: str = './save/train',
                 eval_summary_dir: str = './save/eval',
                 train_checkpoint_dir: str = './save/checkpoint/train',
                 policy_checkpoint_dir: str = './save/checkpoint/policy',
                 replay_checkpoint_dir: str = './save/checkpoint/replay',
                 eval_policy_save_dir: str = './save/eval_policy',
                 collect_episodes_per_iter: int = 1,
                 train_rounds_per_iter: int = 4,
                 eval_interval: int = 10,
                 eval_episodes: int = 1,
                 max_iterations: int = 100000,
                 parallel_num: int = 1,
                 using_py: bool = False
                 ):
        super(LocalTrainer, self).__init__(max_iterations=max_iterations)

        self._eval_interval = eval_interval
        self._train_rounds_one_iter = train_rounds_per_iter
        self._using_py = using_py
        self._eval_policy_save_dir = eval_policy_save_dir

        env_wrappers = env_wrappers or []
        eval_observer = eval_observer or []

        collect_tfa_env, eval_tfa_env = self._construct_tfa_env(env_constructor, env_wrappers, parallel_num)

        if not using_py:
            collect_tfa_env, eval_tfa_env = TFPyEnvironment(collect_tfa_env), TFPyEnvironment(eval_tfa_env)
        self._collect_env, self._eval_env = env_constructor(), env_constructor()

        self._agent = agent
        self._agent.initialize()

        if replay_buffer is None:
            if using_py:
                replay_buffer = PyUniformReplayBuffer(
                    data_spec=tensor_spec.from_spec(agent.collect_data_spec),
                    capacity=replay_cap
                )
            else:
                replay_buffer = TFUniformReplayBuffer(
                    data_spec=tensor_spec.from_spec(agent.collect_data_spec),
                    batch_size=collect_tfa_env.batch_size,
                    max_length=replay_cap
                )
        self._replay = replay_buffer
        self._dataset_iter = iter(self._replay.as_dataset(
            sample_batch_size=batch_size,
            num_steps=2
        ))

        if replay_observer is None:
            replay_observer = [self._replay.add_batch]

        self._train_summary_writer = tf.summary.create_file_writer(
            logdir=train_summary_dir,
            flush_millis=summaries_flush_secs * 1000
        )

        self._eval_summary_writer = tf.summary.create_file_writer(
            logdir=eval_summary_dir,
            flush_millis=summaries_flush_secs * 1000
        )

        self._train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=metric_buffer_num, batch_size=collect_tfa_env.batch_size),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=metric_buffer_num, batch_size=collect_tfa_env.batch_size),
        ] if not using_py else [_SummaryMetric(global_step, self._train_summary_writer)]

        self._eval_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=1, batch_size=eval_tfa_env.batch_size),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=1, batch_size=eval_tfa_env.batch_size),
        ] if not using_py else [_SummaryMetric(global_step, self._eval_summary_writer)]

        self._checkpointers = []

        train_checkpointer = Checkpointer(
            ckpt_dir=train_checkpoint_dir,
            max_to_keep=10,
            agent=agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(self._train_metrics, 'train_metrics')
        )
        self._checkpointers.append(train_checkpointer)

        if not using_py:
            policy_checkpointer = Checkpointer(
                ckpt_dir=policy_checkpoint_dir,
                max_to_keep=10,
                policy=eval_policy,
                global_step=global_step,
                tf_metrics=metric_utils.MetricsGroup(self._eval_metrics, 'eval_metrics')
            )
            self._checkpointers.append(policy_checkpointer)

        if replay_checkpointer is None:
            replay_checkpointer = Checkpointer(
                ckpt_dir=replay_checkpoint_dir,
                max_to_keep=2,
                replay_buffer=replay_buffer
            )
        self._checkpointers.append(replay_checkpointer)

        for checkpointer in self._checkpointers:
            checkpointer.initialize_or_restore()

        self._global_step = global_step

        if initial_collect_policy is not None:
            initial_collect_driver = _make_driver(
                env=collect_tfa_env,
                policy=initial_collect_policy,
                num_episodes=initial_collect_episodes,
                observers=replay_observer + self._train_metrics,
                train_step=global_step,
                using_py=using_py
            )

            initial_collect_driver.run()

        self._collect_driver = _make_driver(
            env=collect_tfa_env,
            policy=collect_policy,
            num_episodes=collect_episodes_per_iter,
            observers=replay_observer + self._train_metrics,
            train_step=global_step,
            using_py=using_py
        )

        self._eval_policy = eval_policy
        self._eval_policy_saver = PolicySaver(eval_policy)
        self._eval_driver = _make_driver(
            env=eval_tfa_env,
            policy=eval_policy,
            num_episodes=eval_episodes,
            observers=eval_observer + self._eval_metrics,
            train_step=global_step,
            using_py=using_py
        )

        self._eval_step_and_summary()

    @property
    def eval_policy(self):
        return self._eval_policy

    @classmethod
    def _construct_tfa_env(cls, env_constructor, env_wrappers, parallel_num):
        def parallel_constructor():
            env = env_constructor()
            for wrapper in env_wrappers:
                env = wrapper(env)
            return env

        if parallel_num > 1:
            collect_tfa_env, eval_tfa_env = ParallelPyEnvironment(
                env_constructors=[parallel_constructor] * parallel_num,
                start_serially=False
            ), parallel_constructor()
        else:
            collect_tfa_env, eval_tfa_env = parallel_constructor(), parallel_constructor()

        return collect_tfa_env, eval_tfa_env

    def _eval_step_and_summary(self):
        self._eval_driver.run()

        with self._eval_summary_writer.as_default():
            if not self._using_py:
                for eval_metric in self._eval_metrics:
                    eval_metric.tf_summaries(train_step=self._global_step)

                metric_utils.log_metrics(self._eval_metrics)

    def _iter_one_step(self, cur_step):
        self._collect_driver.run()

        with self._train_summary_writer.as_default():
            for r in range(self._train_rounds_one_iter):
                experience, _ = next(self._dataset_iter)
                self._agent.train(experience)

            if not self._using_py:
                for train_metric in self._train_metrics:
                    train_metric.tf_summaries(train_step=self._global_step)

        if cur_step % self._eval_interval == 0:
            self._eval_step_and_summary()
            self._eval_policy_saver.save(self._eval_policy_save_dir)
            for checkpointer in self._checkpointers:
                checkpointer.save(global_step=self._global_step.numpy())
