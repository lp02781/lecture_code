from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import copy

from cpprb import ReplayBuffer

import tensorflow as tf
from tensorflow.keras.layers import Dense

from hyperparameter import hp
from environment import Environment
from sac import SAC

if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)

def get_default_rb_dict(size, env):
    return {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {
                "shape": hp.state_dim},
            "next_obs": {
                "shape": hp.state_dim},
            "act": {
                "shape": hp.action_dim},
            "rew": {},
            "done": {}}}

def get_replay_buffer(policy, env, size=None):
    if policy is None or env is None:
        return None

    kwargs = get_default_rb_dict(policy.memory_capacity, env)

    return ReplayBuffer(**kwargs)

class Trainer:
    def __init__(self, policy, env):
        self._policy = policy
        self._env = env

        # experiment settings
        self._max_steps = hp.max_steps
        self._episode_max_steps = hp.episode_max_steps
        self._save_model_interval = hp.save_model_interval
        self._save_summary_interval = hp.save_summary_interval 

        # tensorboard
        self.log_dir = 'logs/'
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.success_rate = tf.keras.metrics.Mean('success', dtype=tf.float32)

    def __call__(self):
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0
        n_training = 0

        random_action_prob = hp.random_action_prob

        replay_buffer = get_replay_buffer(
            self._policy, self._env)

        obs = self._env.reset()

        local_memory = []

        h_score = []
        h_success = []

        while total_steps < self._max_steps:
            if total_steps < self._policy.n_warmup:
                action = 2.0 * np.random.rand(hp.action_dim) - 1.0
                action_length = np.linalg.norm(action)
                if action_length > 1.0:
                    action = action / action_length
            else:
                if np.random.rand() < random_action_prob:
                    action = 2.0 * np.random.rand(hp.action_dim) - 1.0
                    action_length = np.linalg.norm(action)
                    if action_length > 1.0:
                        action = action / action_length
                else:
                    action = self._policy.get_action(obs)

            next_obs, reward, done, _ = self._env.step(action)


            episode_steps += 1
            episode_return += reward
            total_steps += 1
            if episode_steps == self._episode_max_steps:
                done = True

            replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done)
            local_memory.append((obs, action, reward, next_obs, done))

            obs = next_obs

            if done:

                for h in range(episode_steps):
                    state, action, reward, next_state, done = copy.deepcopy(local_memory[h])

                    for her in range(4):
                        future = np.random.randint(h, episode_steps)

                        _, _, _, g, _ = copy.deepcopy(local_memory[future])
                        state[:, 2:4] = g[:, 0:2]
                        if np.linalg.norm(state[:, 0:2] - state[:, 2:4]) <= self._env.goal_bound:
                            continue
                        next_state[:, 2:4] = g[:, 0:2]
                        goal_d = np.linalg.norm(next_state[:, 0:2] - next_state[:, 2:4])
                        if goal_d <= self._env.goal_bound:
                            reward = 0.0
                            done = True
                        else:
                            reward = -1.0
                            done = False
                        replay_buffer.add(obs=state, act=action, next_obs=next_state, rew=reward, done=done)

                local_memory = []


                success = self._env.success
                h_score.append(episode_return)
                h_success.append(success)


                obs = self._env.reset()

                n_episode += 1
                fps = episode_steps / (time.perf_counter() - episode_start_time)
                print("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                    n_episode, total_steps, episode_steps, episode_return, fps))

                if ((n_episode + 1) % 10) == 0 :
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('reward', np.mean(h_score), step=n_episode)
                        tf.summary.scalar('success', np.mean(h_success), step=n_episode)
                    h_score = []
                    h_success = []

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

                if total_steps > self._policy.n_warmup:
                    n_training += 1
                    samples = replay_buffer.sample(self._policy.batch_size)
                    actor_loss, critic_loss, _ = self._policy.train(samples["obs"], samples["act"], samples["next_obs"],
                                                                    samples["rew"],
                                                                    np.array(samples["done"], dtype=np.float32))

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('actor loss', actor_loss, step=n_training)
                        tf.summary.scalar('critic loss', critic_loss, step=n_training)


            if total_steps < self._policy.n_warmup:
                continue


            if n_episode % self._save_model_interval == 0 :
                self._policy._save_model()

if __name__ == '__main__':

    env = Environment()
    policy = SAC()
    trainer = Trainer(policy, env)
    trainer()
