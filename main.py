import agent as agnt
import tqdm
import numpy as np
import pickle
from env import TradingEnv

## main loop settings ##
EPISODES = 5
AGGREGATE_STATS_EVERY = 1
BEST_REWARD = -np.inf

## agent settings ##
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 5_00
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
EPSILON = 1
MIN_EPSILON = 0.001
LR = 0.001
MODEL_NAME = 'test_model'
CUSTOM_TB = True

## env settings ##
MAXIMUM_STEPS = 1
INITIAL_BALANCE = 10_000
TRANSACTION_FEE = 2
RENDER_INTERVAL = 250
LOOKBACK_WINDOW = 24
CANDLE_LENGTH = '1h'

history = {'ep_rewards': [], 'avg_ep_rewards': [],
           'min_ep_rewards': [], 'max_ep_rewards': [], 'eps_history': []}
env = TradingEnv(maximum_steps=MAXIMUM_STEPS, initial_balance=INITIAL_BALANCE,
                 transaction_fee=TRANSACTION_FEE, render_interval=RENDER_INTERVAL, lookback_window=LOOKBACK_WINDOW,
                 candle_length=CANDLE_LENGTH)
agent = agnt.DQNAgent(env, replay_memory_size=REPLAY_MEMORY_SIZE,
                      min_replay_memory_size=MIN_REPLAY_MEMORY_SIZE, minibatch_size=MINIBATCH_SIZE, discount=DISCOUNT,
                      update_target_every=UPDATE_TARGET_EVERY,
                      epsilon=EPSILON, min_epsilon=MIN_EPSILON, lr=LR, model_name=MODEL_NAME, custom_tb=CUSTOM_TB)

for episode in tqdm.tqdm(range(1, EPISODES+1)):
    episode_reward = 0
    step = 1
    current_state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(current_state)
        new_state, reward, done = env.step(action, current_state, episode)
        episode_reward += reward
        agent.update_replay_memory(
            (current_state, action, reward, new_state, done))
        agent.train(done)
        current_state = new_state
        step += 1
    history['ep_rewards'].append(episode_reward)
    history['eps_history'].append(agent.epsilon)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        avg_ep_rewards = np.mean(
            history['ep_rewards'][-AGGREGATE_STATS_EVERY:])
        history['avg_ep_rewards'].append(avg_ep_rewards)
        min_ep_rewards = np.min(history['ep_rewards'][-AGGREGATE_STATS_EVERY:])
        history['min_ep_rewards'].append(min_ep_rewards)
        max_ep_rewards = np.max(history['ep_rewards'][-AGGREGATE_STATS_EVERY:])
        history['max_ep_rewards'].append(max_ep_rewards)
        if CUSTOM_TB:
            agent.tensorboard.update_stats(
                reward_avg=avg_ep_rewards, reward_min=min_ep_rewards, reward_max=max_ep_rewards, epsilon=agent.epsilon)
        print(f'Episode: {episode}, Episode_reward: {episode_reward}, Average_reward: {avg_ep_rewards}, Min_reward: {min_ep_rewards}, Max_reward: {max_ep_rewards}, Epsilon: {agent.epsilon}')
        if BEST_REWARD < max_ep_rewards:
            BEST_REWARD = max_ep_rewards
            agent.model.save(
                f'./models/{MODEL_NAME}/model.h5')
    # Choose one epsilon decaying scheme
    # agent.decay_epsilon()
    agent.epsilon = agent.multiplicative_exp_decay_epsilon(EPSILON, episode-1)

with open(f'./models/{MODEL_NAME}/history.pkl', 'wb') as f:
    pickle.dump(history, f)

with open(f'./models/{MODEL_NAME}/action_history.pkl', 'wb') as f:
    pickle.dump(env.action_history, f)
