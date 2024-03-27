import numpy as np
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import tensorflow as tf

HIDDEN_SIZE = 32


class ActorNetwork(object):
    def __init__(self, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, _, _ = self.create_actor_network(state_size, action_size)

        self.optimizer = Adam(learning_rate=LEARNING_RATE)
        self.target_train(TAU=1)

    @tf.function
    def compute_gradients(self, states, action_grads):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss = -tf.reduce_sum(predictions * action_grads)

        gradients = tape.gradient(loss, self.weights)
        return gradients

    def train(self, states, action_grads):
        gradients = self.compute_gradients(states, action_grads)
        self.optimizer.apply_gradients(zip(gradients, self.weights))

    def target_train(self,TAU=None):
        if TAU is None:
            TAU=self.TAU
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = TAU * actor_weights[i] + (1 - TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def load_model_weights(self, weight_file):
        self.model.load_weights(weight_file)

    def save_model_weights(self, suffix1,suffix2):
        self.model.save_weights(suffix1)
        self.target_model.save_weights(suffix2)

    def create_actor_network(self, state_size, action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        sequence = Sequential([
            Dense(HIDDEN_SIZE, activation='relu'),  # First hidden layer
            Dense(HIDDEN_SIZE, activation='relu'),  # Second hidden layer
            Dense(HIDDEN_SIZE, activation='relu'),  # Third hidden layer
            Dense(action_dim, activation='sigmoid'),  # Output layer [0, 1], Deterministic Policy
        ])  # actor_model
        V = sequence(S)
        model = Model(S, V)
        print(f"model.trainable weights {model.trainable_weights}")
        weights_values = model.get_weights()
        trainable_weights = model.trainable_weights
        for weight, value in zip(trainable_weights, weights_values):
            print(f"Weight shape: {weight.shape}, Value: {value}")
        return model, model.trainable_weights, S
