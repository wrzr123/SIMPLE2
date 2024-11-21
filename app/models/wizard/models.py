import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout, Lambda, Add
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.distributions import CategoricalProbabilityDistribution

ACTIONS = 4 + 16 + 15
FEATURE_SIZE = 256
DEPTH = 5
VALUE_DEPTH = 1
POLICY_DEPTH = 1

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        ### From ChatGPT
        # with tf.variable_scope("model", reuse=reuse):
        #     # Create a simple feedforward neural network for processing card observations
        #     flattened_input = Flatten()(self.processed_obs)
        #
        #     # Hidden layer 1
        #     dense_1 = Dense(64, activation='relu')(flattened_input)
        #     dense_1 = BatchNormalization()(dense_1)
        #     dense_1 = Dropout(0.3)(dense_1)
        #
        #     # Hidden layer 2
        #     dense_2 = Dense(64, activation='relu')(dense_1)
        #     dense_2 = BatchNormalization()(dense_2)
        #     dense_2 = Dropout(0.3)(dense_2)
        #
        #     # Policy head (probabilities for each action)
        #     self._policy = Dense(ac_space.n, activation=None)(dense_2)
        #
        #     # Value head for critic
        #     self._value_fn = Dense(1, activation=None)(dense_2)
        #
        #     # Create the probability distribution
        #     self._proba_distribution = CategoricalProbabilityDistribution(self._policy)
        #
        # self._setup_init()

        ### Copied from butterfly
        with tf.variable_scope("model", reuse=reuse):

            obs, legal_actions = split_input(self.processed_obs, ACTIONS)

            extracted_features = resnet_extractor(obs, **kwargs)

            self._policy = policy_head(extracted_features, legal_actions)
            self._value_fn, self.q_value = value_head(extracted_features)
            self._proba_distribution  = CategoricalProbabilityDistribution(self._policy)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


def value_head(y):
    for _ in range(VALUE_DEPTH):
        y = dense(y, FEATURE_SIZE)
    vf = dense(y, 1, batch_norm=False, activation='tanh', name='vf')
    q = dense(y, ACTIONS, batch_norm=False, activation='tanh', name='q')
    return vf, q


def policy_head(y, legal_actions):
    for _ in range(POLICY_DEPTH):
        y = dense(y, FEATURE_SIZE)
    policy = dense(y, ACTIONS, batch_norm=False, activation=None, name='pi')

    mask = Lambda(lambda x: (1 - x) * -1e8)(legal_actions)

    policy = Add()([policy, mask])
    return policy


def dense(y, filters, batch_norm = False, activation = 'relu', name = None):

    if batch_norm or activation:
        y = Dense(filters)(y)
    else:
        y = Dense(filters, name = name)(y)

    if batch_norm:
        if activation:
            y = BatchNormalization(momentum = 0.9)(y)
        else:
            y = BatchNormalization(momentum = 0.9, name = name)(y)

    if activation:
        y = Activation(activation, name = name)(y)

    return y

def split_input(obs, split):
    return   obs[:,:-split], obs[:,-split:]

def resnet_extractor(y, **kwargs):
    y = dense(y, FEATURE_SIZE)
    for _ in range(DEPTH):
        y = residual(y, FEATURE_SIZE)

    return y

def residual(y, filters):
    shortcut = y

    y = dense(y, filters)
    y = dense(y, filters, activation = None)
    y = Add()([shortcut, y])
    y = Activation('relu')(y)

    return y