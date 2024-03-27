import numpy as np
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN_SIZE = 32

class CriticNetwork(object):
    def __init__(self, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, _, _ = self.create_critic_network(state_size, action_size)
        self.model.compile(optimizer=Adam(learning_rate=self.LEARNING_RATE), loss='mse')
        self.target_train(TAU=1)

        # self.optimizer = Adam(learning_rate=LEARNING_RATE)
        # self.loss_object = tf.keras.losses.MeanSquaredError()

    def gradients(self, states, actions):
        with tf.GradientTape() as tape:
            # Convert NumPy arrays to TensorFlow tensors
            states_tensor = tf.convert_to_tensor(states)
            actions_tensor = tf.convert_to_tensor(actions)
            tape.watch(actions_tensor)
            Q_values = self.model([states_tensor, actions_tensor], training=True)
            # print(f'Q-values: {Q_values}')
            # print(f"actions tensor {actions_tensor}")
            return tape.gradient(Q_values, actions_tensor)

    def target_train(self,TAU=None):
        if TAU is None:
            TAU=self.TAU
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = TAU * critic_weights[i] + (1 - TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def save_model_weights(self, suffix1,suffix2):
        # Helper function to save your model / weights.
        self.model.save_weights(suffix1)
        self.target_model.save_weights(suffix2)

    def load_model_weights(self, weight_file):
        # Helper function to load model weights.
        self.model.load_weights(weight_file)

    def create_critic_network(self, state_size, action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        concat_input = Concatenate(axis=-1)([S, A])
        sequence = Sequential([
            Dense(HIDDEN_SIZE, activation='relu'),  # First hidden layer
            Dense(HIDDEN_SIZE, activation='relu'),  # Second hidden layer
            Dense(HIDDEN_SIZE, activation='relu'),  # Third hidden layer
            Dense(1, activation='linear'),  # Output layer
        ])  # critic_model
        sequence_out = sequence(concat_input)
        model = Model([S, A], sequence_out)
        return model, A, S
