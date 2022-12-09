from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import copy

import tensorflow as tf
from tensorflow.keras.layers import Dense

from hyperparameter import hp

class CriticV(tf.keras.Model):
    def __init__(self, state_shape, name='vf'):
        super().__init__(name=name)

        self.l1 = Dense(hp.hidden_layer, name="L1", activation='relu')
        self.l2 = Dense(hp.hidden_layer, name="L2", activation='relu')
        self.l3 = Dense(1, name="L3", activation='linear')


    def call(self, states):
        features = self.l1(states)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1, name="values")

class CriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, name='qf'):
        super().__init__(name=name)

        self.l1 = Dense(hp.hidden_layer, name="L1", activation='relu')
        self.l2 = Dense(hp.hidden_layer, name="L2", activation='relu')
        self.l3 = Dense(1, name="L2", activation='linear')


    def call(self, inputs):
        [states, actions] = inputs
        features = tf.concat([states, actions], axis=1)
        features = self.l1(features)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1)
