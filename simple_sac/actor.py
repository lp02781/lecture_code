from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import copy

import tensorflow as tf
from tensorflow.keras.layers import Dense

from hyperparameter import hp

class DiagonalGaussian():
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def likelihood_ratio(self, x, old_param, new_param):
        llh_new = self.log_likelihood(x, new_param)
        llh_old = self.log_likelihood(x, old_param)
        return tf.math.exp(llh_new - llh_old)

    def log_likelihood(self, x, param):
        """
        Compute log likelihood as:
        -N/2log(2*pi*sigma^2)-1/(2*sigma^2) * Sum(x - mu)^2
        """
        means = param["mean"]
        log_stds = param["log_std"]
        assert means.shape == log_stds.shape
        zs = (x - means) / tf.exp(log_stds)
        return - tf.reduce_sum(log_stds, axis=-1) \
               - 0.5 * tf.reduce_sum(tf.square(zs), axis=-1) \
               - 0.5 * self.dim * tf.math.log(2 * np.pi)

    def sample(self, param):
        means = param["mean"]
        log_stds = param["log_std"]
        # reparameterization
        return means + tf.random.normal(shape=means.shape) * tf.math.exp(log_stds)

    def entropy(self, param):
        log_stds = param["log_std"]
        return tf.reduce_sum(log_stds + tf.math.log(tf.math.sqrt(2 * np.pi * np.e)), axis=-1)


class GaussianActor(tf.keras.Model):
    LOG_SIG_CAP_MAX = hp.LOG_SIG_CAP_MAX
    LOG_SIG_CAP_MIN = hp.LOG_SIG_CAP_MIN
    EPS = hp.EPS

    def __init__(self, state_shape, action_dim, max_action, hidden_activation="relu",
                 tanh_mean=False, tanh_std=False,
                 fix_std=False, const_std=0.1,
                 state_independent_std=False, name='GaussianPolicy'):
        super().__init__(name=name)
        self.dist = DiagonalGaussian(dim=action_dim)
        self._fix_std = fix_std
        self._tanh_std = tanh_std
        self._const_std = const_std
        self._max_action = max_action
        self._state_independent_std = state_independent_std

        self.l1 = Dense(hp.hidden_layer, name="L1", activation=hidden_activation)
        self.l2 = Dense(hp.hidden_layer, name="L2", activation=hidden_activation)
        self.out_mean = Dense(action_dim, name="L_mean", activation='tanh')
        activation = 'tanh' if tanh_std else None
        self.out_log_std = Dense(action_dim, name="L_sigma", activation=activation)


    def _compute_dist(self, states):
        features = self.l1(states)
        features = self.l2(features)
        mean = self.out_mean(features)
        log_std = self.out_log_std(features)
        log_std = tf.clip_by_value(log_std, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)

        return {"mean": mean, "log_std": log_std}

    def call(self, states, test=False):
        param = self._compute_dist(states)
        if test:
            raw_actions = param["mean"]
        else:
            raw_actions = self.dist.sample(param)
        logp_pis = self.dist.log_likelihood(raw_actions, param)

        actions = raw_actions

        return actions * self._max_action, logp_pis, param
