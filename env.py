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
        self.action_space = np.arange(0,9)
        # 5 equals the OHCLV data
        self.observation_space = (self.lookback_window, 5)
        self.maximum_steps = maximum_steps
        self.render_interval = render_interval
        self.action_list = []
        self.action_history = {}
        self.balance_dict = {'Wallet_balance': [], 'Balance': [], 'Current_price': []}
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
        OHCLV_features = np.array(
            df.loc[df.index[0+step]:df.index[self.lookback_window-1+step]][self.X_COLS])
        external_balance_features = np.array([self.balance, self.wallet_balance])
        state = (OHCLV_features, external_balance_features)
        return state

    def perform_action(self, action, current_price):
        borrow_transaction_fee = False
        if self.balance < self.transaction_fee and action in [5,6,7,8]:
            borrow_transaction_fee = True
            # print(Enable bot to sell Crypto when having zero fiat balance)
            amount_borrowed = self.transaction_fee - self.balance
            self.balance += amount_borrowed
        if action == 0:
            # print('Action 0')
            pass
        elif action in [1,2,3,4]:
            if self.balance > self.transaction_fee:
                # print(f'Action [1,2,3,4]: Bought Crypto at a current price of {current_price}')
                amount_crypto_bought =((0.25 * action * self.balance) - self.transaction_fee)/current_price
                self.balance -= amount_crypto_bought*current_price + self.transaction_fee
                self.wallet_balance += amount_crypto_bought
            elif self.balance == 0:
                action += 10 # Just for evaluation purposes do differ these events 
                # print('Action [1,2,3,4] has been choosen but has no effect since there is no fiat money left to buy')
                pass
        elif action in [5,6,7,8] and self.balance >= self.transaction_fee:
            # print(f'Action [5,6,7,8]: Sold Crypto at a current price of {current_price}')
            amount_crypto_sold = 0.25 * (action-4) * self.wallet_balance
            self.balance += amount_crypto_sold*current_price - self.transaction_fee
            self.wallet_balance -= amount_crypto_sold
        else:
            print('Start')
            print('Action 9: None of the other actions was possible (Should not appear and therefore should be fixed)')
            self.render(current_price)
            print(f'Action {action}')
            print('End')
            action = 9
        if borrow_transaction_fee:
            # print('Return borrowed transaction fee')
            self.balance -= amount_borrowed
        self.action_list.append(action)
        self.balance_dict['Wallet_balance'].append(self.wallet_balance)
        self.balance_dict['Balance'].append(self.balance)
        self.balance_dict['Current_price'].append(current_price)

    def update_networth(self, current_price):
        self.net_worth = self.balance + self.wallet_balance*current_price

    def render(self, current_price):
        print(f'Step: {self.episode_step}, Net_worth: {self.net_worth}, Crypto_balance: {self.wallet_balance} BTC, Fiat_balance: {self.balance} USD, BTC_Price: {current_price} USD')

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.wallet_balance = 0
        self.episode_step = 0
        self.action_list = []
        self.balance_dict = {'Wallet_balance': [], 'Balance': [], 'Current_price': []}
        current_state = self.get_observation(self.df, self.episode_step)
        return current_state

    def step(self, action, current_state, episode_counter):
        current_networth = self.net_worth
        current_price = current_state[0][-1][3] # always take closing price of last candle
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

        next_observation = self.get_observation(self.df, self.episode_step)
        if self.episode_step % self.render_interval == 0:
            self.render(current_price)

        return next_observation, reward, done
