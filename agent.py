import tensorflow as tf
import numpy as np
import random
import custom_tensorboard as ctb
import time
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from collections import deque

print('Using TensorFlow {:s} with {:d} GPUs'.format(
    tf.__version__, len(tf.config.experimental.list_physical_devices('GPU'))))


class DQNAgent:
    def __init__(self, env, replay_memory_size, min_replay_memory_size, minibatch_size, discount, update_target_every, epsilon, min_epsilon, lr, model_name, custom_tb=False):
        self.env = env
        self.replay_memory_size = replay_memory_size
        self.min_replay_memory_size = min_replay_memory_size
        self.minibatch_size = minibatch_size
        self.discount = discount
        self.update_target_every = update_target_every
        self.model_name = model_name
        self.lr = lr
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.target_update_counter = 0
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        if custom_tb:
            self.tensorboard = ctb.ModifiedTensorBoard(
                log_dir=f'./logs/{model_name}-{int(time.time())}')

    def create_model(self):
        model = Sequential()
        model.add(LSTM(16, return_sequences=True,
                  input_shape=self.env.observation_space))
        model.add(LSTM(16, return_sequences=True))
        model.add(LSTM(8, return_sequences=True))
        model.add(LSTM(8))
        model.add(Dense(len(self.env.action_space), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.env.action_space)
        else:
            action = np.argmax(self.model.predict(
                np.array(state).reshape(-1, *self.env.observation_space), verbose=0))
        return action
    
    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)

    def multiplicative_exp_decay_epsilon(self, c0, iteration, alpha=0.95):
        c = max(self.min_epsilon, c0 * alpha**iteration)
        return c

    def train(self, terminal_state):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        current_states = np.array([transition[0] for transition in minibatch])
        q_values_list = self.model.predict(current_states, verbose=0)

        future_states = np.array([transition[3] for transition in minibatch])
        future_q_values_list = self.target_model.predict(
            future_states, verbose=0)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_q_values_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_q_values = q_values_list[index]
            current_q_values[action] = new_q

            X.append(current_state)
            y.append(current_q_values)

        self.model.fit(np.array(X), np.array(
            y), batch_size=self.minibatch_size, verbose=0, shuffle=False)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
