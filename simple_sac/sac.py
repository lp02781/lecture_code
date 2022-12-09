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
from actor import GaussianActor
from critic import CriticV, CriticQ

class SAC():
    def __init__(self):
        super().__init__()

        gpu = 0
        max_action = 1.
        state_shape = hp.state_dim #env.observation_space.shape
        action_dim = hp.action_dim #env.action_space.high.size

        lr = hp.lr
        sigma = hp.sigma
        tau = hp.tau
        self.state_ndim = len((1, state_shape))

        self.policy_name = hp.policy_name
        self.update_interval = hp.update_interval
        self.batch_size = hp.batch_size
        self.discount = hp.discount
        self.n_warmup = hp.n_warmup
        self.max_grad = hp.max_grad
        self.memory_capacity = hp.memory_capacity
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"


        # Define and initialize Actor network
        self.actor = GaussianActor(state_shape, action_dim, max_action, tanh_mean=False, tanh_std=False)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Define and initialize Critic network
        self.qf1 = CriticQ(state_shape, action_dim, name="qf1")
        self.qf2 = CriticQ(state_shape, action_dim, name="qf2")
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.vf = CriticV(state_shape)
        self.vf_target = CriticV(state_shape)
        update_target_variables(self.vf_target.weights,
                                self.vf.weights, tau=1.)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


        # Set hyper-parameters
        self.sigma = sigma
        self.tau = tau

        self.log_alpha = tf.Variable(0., dtype=tf.float32)
        self.alpha = tf.Variable(0., dtype=tf.float32)
        self.alpha.assign(tf.exp(self.log_alpha))
        self.target_alpha = -action_dim
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _save_model(self):
        self.actor.save_weights('./save/actor', save_format='tf')
        self.qf1.save_weights('./save/qf1', save_format='tf')
        self.qf2.save_weights('./save/qf2', save_format='tf')
        self.vf.save_weights('./save/vf', save_format='tf')
        self.vf_target.save_weights('./save/vf_target', save_format='tf')

    def _load_model(self):
        self.actor.load_weights('./save/actor')
        self.actor_target.load_weights('./save/actor_target')
        self.critic.load_weights('./save/critic')
        self.critic_target.load_weights('./save/critic_target')
        self.critic_2.load_weights('./save/critic_2')
        self.critic_2_target.load_weights('./save/critic_2_target')

    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim

        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(tf.constant(state), test)

        return action.numpy()[0] if is_single_state else action

    @tf.function
    def _get_action_body(self, state, test):
        return self.actor(state, test)[0]

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)
        td_errors, actor_loss, vf_loss, qf_loss, logp_min, logp_max, logp_mean = \
            self._train_body(states, actions, next_states,
                             rewards, done, weights)

        return actor_loss, qf_loss, td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        with tf.device(self.device):
            if tf.rank(rewards) == 2:
                rewards = tf.squeeze(rewards, axis=1)
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Compute loss of critic Q
                current_q1 = self.qf1([states, actions])
                current_q2 = self.qf2([states, actions])
                vf_next_target = self.vf_target(next_states)

                target_q = tf.stop_gradient(
                    rewards + not_dones * self.discount * vf_next_target)

                td_loss_q1 = tf.reduce_mean(huber_loss(
                    target_q - current_q1, delta=self.max_grad) * weights)
                td_loss_q2 = tf.reduce_mean(huber_loss(
                    target_q - current_q2, delta=self.max_grad) * weights)  # Eq.(7)

                # Compute loss of critic V
                current_v = self.vf(states)

                sample_actions, logp, _ = self.actor(states)  # Resample actions to update V
                current_q1 = self.qf1([states, sample_actions])
                current_q2 = self.qf2([states, sample_actions])
                current_min_q = tf.minimum(current_q1, current_q2)

                target_v = tf.stop_gradient(
                    current_min_q - self.alpha * logp)
                td_errors = target_v - current_v
                td_loss_v = tf.reduce_mean(
                    huber_loss(td_errors, delta=self.max_grad) * weights)  # Eq.(5)

                # Compute loss of policy
                policy_loss = tf.reduce_mean(
                    (self.alpha * logp - current_min_q) * weights)  # Eq.(12)

                alpha_loss = -tf.reduce_mean(
                    (self.log_alpha * tf.stop_gradient(logp + self.target_alpha)))

            q1_grad = tape.gradient(td_loss_q1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss_q2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self.qf2.trainable_variables))

            vf_grad = tape.gradient(td_loss_v, self.vf.trainable_variables)
            self.vf_optimizer.apply_gradients(
                zip(vf_grad, self.vf.trainable_variables))
            update_target_variables(
                self.vf_target.weights, self.vf.weights, self.tau)

            actor_grad = tape.gradient(
                policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(
                zip(alpha_grad, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))

            del tape

        return td_errors, policy_loss, td_loss_v, td_loss_q1, tf.reduce_min(logp), tf.reduce_max(logp), tf.reduce_mean(
            logp)
            
def huber_loss(x, delta=1.):
    delta = tf.ones_like(x) * delta
    less_than_max = 0.5 * tf.square(x)
    greater_than_max = delta * (tf.abs(x) - 0.5 * delta)
    return tf.where(
        tf.abs(x) <= delta,
        x=less_than_max,
        y=greater_than_max)
        
def update_target_variables(target_variables,
                            source_variables,
                            tau=1.0,
                            use_locking=False,
                            name="update_target_variables"):

    if not isinstance(tau, float):
        raise TypeError("Tau has wrong type (should be float) {}".format(tau))
    if not 0.0 < tau <= 1.0:
        raise ValueError("Invalid parameter tau {}".format(tau))
    if len(target_variables) != len(source_variables):
        raise ValueError("Number of target variables {} is not the same as "
                         "number of source variables {}".format(
                             len(target_variables), len(source_variables)))

    same_shape = all(trg.get_shape() == src.get_shape()
                     for trg, src in zip(target_variables, source_variables))
    if not same_shape:
        raise ValueError("Target variables don't have the same shape as source "
                         "variables.")

    def update_op(target_variable, source_variable, tau):
        if tau == 1.0:
            return target_variable.assign(source_variable, use_locking)
        else:
            return target_variable.assign(
                tau * source_variable + (1.0 - tau) * target_variable, use_locking)

    # with tf.name_scope(name, values=target_variables + source_variables):
    update_ops = [update_op(target_var, source_var, tau)
                  for target_var, source_var
                  in zip(target_variables, source_variables)]
    return tf.group(name="update_all_variables", *update_ops)

