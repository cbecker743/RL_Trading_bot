import data_loader as dl
import pandas as pd
import numpy as np
import random


class TradingEnv:
    def __init__(self, maximum_steps, initial_balance, transaction_fee, render_interval, lookback_window, candle_length, load_new_data=False):
        self.load_new_data = load_new_data
        self.candle_length = candle_length
        # only necessary to calculate prediction targets(not used for RL yet)
        self.future_steps = 1
        self.Y_COLS = ['close_target']
        self.X_COLS = ['open', 'high', 'low', 'close', 'volume']
        self.df = self.read_df()
        self.lookback_window = lookback_window
        self.episode_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.wallet_balance = 0
        self.net_worth = self.balance + self.wallet_balance
        self.transaction_fee = transaction_fee
        self.action_space = np.array([0, 1, 2])
        # 5 equals the OHCLV data
        self.observation_space = (self.lookback_window, 5)
        self.maximum_steps = maximum_steps
        self.render_interval = render_interval
        self.action_list = []
        self.action_history = {}
        self.balance_dict = {'Wallet_balance': [], 'Balance': []}
        self.balance_history = {}

    def read_df(self):
        if self.load_new_data:
            print(self.load_new_data)
            dl.scrape_candles_to_csv(
                'dataframe.csv', 'binance', 3, 'BTC/USDT', self.candle_length, '2015-01-0100:00:00Z', 1000)
        df = pd.read_csv('./data/Binance/dataframe.csv',
                         header=0, index_col='timestamp')
        df = dl.preprocess_dataset(
            df, self.candle_length, self.future_steps, self.Y_COLS)
        return df

    def get_observation(self, df, step):
        state = np.array(
            df.loc[df.index[0+step]:df.index[self.lookback_window-1+step]][self.X_COLS])
        state_target = np.array(
            df.loc[df.index[0+step]:df.index[self.lookback_window-1+step]][self.Y_COLS])
        return state, state_target

    def perform_action(self, action, current_price):
        if action == 0:
            # print('Action 0')
            pass
        elif action == 1 and self.balance > 0:
            # print(f'Action 1: Bought Crypto at current price of {current_price}')
            amount_crypto_bought = random.uniform(0, (self.balance-self.transaction_fee)/current_price)
            self.balance -= amount_crypto_bought*current_price + self.transaction_fee
            self.wallet_balance += amount_crypto_bought
        elif action == 2 and self.wallet_balance > 0:
            # print(f'Action 2: Sold Crypto at current price of {current_price}')
            amount_crypto_sold = random.uniform(0, self.wallet_balance)
            self.balance += amount_crypto_sold*current_price - self.transaction_fee
            self.wallet_balance -= amount_crypto_sold
        self.action_list.append(action)
        self.balance_dict['Wallet_balance'].append(self.wallet_balance)
        self.balance_dict['Balance'].append(self.balance)

    def update_networth(self, current_price):
        self.net_worth = self.balance + self.wallet_balance*current_price

    def render(self):
        print(f'Step: {self.episode_step}, Net_worth: {self.net_worth}, Crypto_balance: {self.wallet_balance} BTC, Fiat_balance: {self.balance} USD')

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.wallet_balance = 0
        self.episode_step = 0
        self.action_list = []
        self.balance_dict = {'Wallet_balance': [], 'Balance': []}
        current_state, _ = self.get_observation(self.df, self.episode_step)
        return current_state

    def step(self, action, current_state, episode_counter):
        current_networth = self.net_worth
        # current_price = random.uniform(
        #     current_state[-1][0], current_state[-1][3])
        current_price = current_state[-1][0] # always take open price of last candle
        self.episode_step += 1
        self.perform_action(action, current_price)
        self.update_networth(current_price)
        reward = self.net_worth - current_networth
        if self.net_worth <= self.initial_balance/2 or self.episode_step >= self.maximum_steps:
            done = True
            self.action_history[f'Episode_{episode_counter}'] = self.action_list
            self.balance_history[f'Episode_{episode_counter}'] = self.balance_dict
        else:
            done = False

        next_observation, _ = self.get_observation(self.df, self.episode_step)
        if self.episode_step % self.render_interval == 0:
            self.render()

        return next_observation, reward, done
