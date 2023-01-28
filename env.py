import data_loader as dl
import pandas as pd
import pandas_ta as ta
import numpy as np
# import helpers


class TradingEnv:
    def __init__(self, maximum_steps, initial_balance, transaction_fee, render_interval,
                 lookback_window, candle_length, add_technical_indicators, add_time_info, start_time, run_mode, load_new_data):
        self.load_new_data = load_new_data
        self.candle_length = candle_length
        # only necessary to calculate prediction targets(not used for RL yet)
        self.episode_start_index = None
        self.future_steps = 1
        self.X_COLS = None
        add_technical_indicators
        self.start_time = start_time
        self.df = self.read_df(add_technical_indicators, add_time_info)
        self.start_index = len(self.df[:self.start_time]) - 1
        self.lookback_window = lookback_window
        self.episode_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.wallet_balance = 0
        self.net_worth = self.balance + self.wallet_balance
        self.transaction_fee = transaction_fee
        self.action_space = np.arange(0, 9)
        # 5 equals the OHCLV data
        self.run_mode = run_mode
        self.observation_space = (self.lookback_window, len(self.X_COLS))
        self.maximum_steps = maximum_steps
        self.render_interval = render_interval
        self.evaluation_dict = {'Wallet_balance': [], 'Balance': [
        ], 'Current_price': [], 'Datetime': [], 'Action': []}
        self.evaluation_dict_history = {}

    def read_df(self, add_technical_indicators, add_time_info):
        if self.load_new_data:
            print('asd')
            dl.scrape_candles_to_csv(
                'dataframe.csv', 'binance', 3, 'BTC/USDT', self.candle_length, '2015-01-0100:00:00Z', 1000)
        df = pd.read_csv(f'./data/Binance/dataframe_{self.candle_length}.csv',
                         header=0, index_col='timestamp')
        df = dl.preprocess_dataset(
            df, self.candle_length, self.future_steps)
        if add_technical_indicators:
            MyStrategy = ta.Strategy(
                name="DCSMA10",
                ta=[
                    {"kind": "rsi"},
                    {"kind": "macd"},
                    {"kind": "stoch"},
                    {"kind": "bbands"},
                    {"kind": "roc"}
                ]
            )
            df.ta.strategy(MyStrategy)
        if add_time_info:
            df['month'] = df.index.month - 1
            df['day_of_week'] = df.index.day % 7
            df['time_of_day'] = df.index.hour
        self.X_COLS = df.columns.tolist()
        df = df.dropna()
        assert(df.index[0] < pd.Timestamp(self.start_time))
        return df

    def get_observation(self, df: pd.DataFrame, step: int) -> tuple:
        if self.run_mode == 'sequential':
            OHCLV_features = np.array(
                df.iloc[self.start_index+step-self.lookback_window:self.start_index+step][self.X_COLS])
            # print(df.iloc[self.start_index+step-self.lookback_window:self.start_index+step][self.X_COLS][-1:].index[0])
            external_balance_features = np.array(
                [self.balance, self.wallet_balance])
            state = (OHCLV_features, external_balance_features)
        elif self.run_mode == 'random':
            OHCLV_features = np.array(
                df.iloc[self.episode_start_index+step-self.lookback_window:self.episode_start_index+step][self.X_COLS])
            # print(df.iloc[self.episode_start_index+step-self.lookback_window:self.episode_start_index+step][self.X_COLS][-1:].index[0])
            external_balance_features = np.array(
                [self.balance, self.wallet_balance])
            state = (OHCLV_features, external_balance_features)
        else:
            run_mode_list = ['sequential', 'random']
            assert(self.run_mode in run_mode_list)
        return state

    def perform_action(self, action, current_price):
        borrow_transaction_fee = False
        if self.balance < self.transaction_fee and action in [5, 6, 7, 8]:
            borrow_transaction_fee = True
            # print(Enable bot to sell Crypto when having zero fiat balance)
            amount_borrowed = self.transaction_fee - self.balance
            self.balance += amount_borrowed
        if action == 0:
            # print('Action 0')
            pass
        elif action in [1, 2, 3, 4]:
            if self.balance > self.transaction_fee:
                # print(f'Action [1,2,3,4]: Bought Crypto at a current price of {current_price}')
                amount_crypto_bought = (
                    (0.25 * action * self.balance) - self.transaction_fee)/current_price
                self.balance -= amount_crypto_bought*current_price + self.transaction_fee
                self.wallet_balance += amount_crypto_bought
            elif self.balance <= self.transaction_fee:
                action += 9  # Just for evaluation purposes do differ these events
                # print('Action [1,2,3,4] has been choosen but has no effect since there is no fiat money left to buy')
                pass
            else:
                print(
                    'Action 9.1: None of the other actions was possible (Should not appear and therefore should be fixed)')
                action = 9
        elif action in [5, 6, 7, 8] and self.balance >= self.transaction_fee:
            # print(f'Action [5,6,7,8]: Sold Crypto at a current price of {current_price}')
            amount_crypto_sold = 0.25 * (action-4) * self.wallet_balance
            self.balance += amount_crypto_sold*current_price - self.transaction_fee
            self.wallet_balance -= amount_crypto_sold
        else:
            print('Action 9.2: None of the other actions was possible (Should not appear and therefore should be fixed)')
            action = 9
        if borrow_transaction_fee:
            # print('Return borrowed transaction fee')
            self.balance -= amount_borrowed
        self.evaluation_dict['Action'].append(action)
        self.evaluation_dict['Wallet_balance'].append(self.wallet_balance)
        self.evaluation_dict['Balance'].append(self.balance)
        self.evaluation_dict['Current_price'].append(current_price)

    def update_networth(self, current_price):
        self.net_worth = self.balance + self.wallet_balance*current_price

    def render(self, current_price):
        print(f'Step: {self.episode_step}, Net_worth: {self.net_worth}, Crypto_balance: {self.wallet_balance} BTC, Fiat_balance: {self.balance} USD, BTC_Price: {current_price} USD')

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.wallet_balance = 0
        self.episode_step = 0
        self.evaluation_dict = {'Wallet_balance': [], 'Balance': [
        ], 'Current_price': [], 'Datetime': [], 'Action': []}
        current_state = self.get_observation(self.df, self.episode_step)
        return current_state

    def retrieve_current_datetime_by_step(self):
        # The current datetime is always 1 step further then the last entry of the current state dataframe, i.e. the current time is always the starting time of the next candle
        if self.run_mode == 'sequential':
            return self.df.iloc[self.start_index+self.episode_step][self.X_COLS].name
        else:
            return self.df.iloc[self.episode_start_index+self.episode_step][self.X_COLS].name

    def step(self, action, current_state, episode_counter):
        current_networth = self.net_worth
        # always take closing price of last candle
        current_price = current_state[0][-1][3]
        current_datetime = self.retrieve_current_datetime_by_step()
        self.evaluation_dict['Datetime'].append(current_datetime)
        self.episode_step += 1
        self.perform_action(action, current_price)
        self.update_networth(current_price)
        reward = self.net_worth - current_networth
        if self.net_worth <= self.initial_balance/2 or self.episode_step >= self.maximum_steps:
            done = True
            self.evaluation_dict_history[f'Episode_{episode_counter}'] = self.evaluation_dict
        else:
            done = False

        next_observation = self.get_observation(self.df, self.episode_step)
        if self.episode_step % self.render_interval == 0:
            self.render(current_price)

        return next_observation, reward, done
