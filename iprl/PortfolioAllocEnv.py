import numpy as np
import pandas as pd
import gym
import matplotlib
import matplotlib.pyplot as plt
from gym.utils import seeding
from gym import spaces
#from stable_baselines.common.base_class import maybe_make_env
from stable_baselines.common.vec_env import DummyVecEnv
import iprl.IPRLConfig as config
import time
matplotlib.use("Agg")


class PortfolioAllocEnv(gym.Env):

    """ Class PortfolioAlloc 

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """

    metadata = {"render.modes": ["human"]}

    def __init__(
                  self,
                  df,
                  stock_dim,
                  hmax,
                  initial_amount,
                  transaction_cost_pct,
                  reward_scaling,
                  state_space,
                  action_space,
                  tech_indicator_list,
                  turbulence_threshold=None,
                  lookback=252,
                  day=0 ):

        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1

        self.day         = day
        self.lookback    = lookback
        self.df          = df
        self.stock_dim   = stock_dim
        self.hmax        = hmax

        self.initial_amount       = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling       = reward_scaling
        self.state_space          = state_space
        self.action_space         = action_space
        self.tech_indicator_list  = tech_indicator_list
        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        
        # Shape = (34, 30)
        # covariance matrix + technical indicators

        self.observation_space = spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.state_space + len(self.tech_indicator_list), self.state_space),
                        )

        # load data from a pandas dataframe
        self.data  = self.df.loc[self.day, :]

        self.covs  = self.data["cov_list"].values[0]

        self.state = np.append(
                       np.array(self.covs),
                       [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                        axis=0,
                       )

        self.terminal = False

        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.portfolio_variance = [0]
        self.portfolio_account  = [[self.initial_amount * 1/action_space]*action_space] 
        self.actions_memory = np.array([[1 / self.stock_dim] * self.stock_dim])
        self.date_memory = [self.data.date.unique()[0]]

        self.sharpe = 0 

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot(df.daily_return.cumsum(), "r")
            cum_rewards_png=f"{config.RESULTS_DIR}/cumulative_reward.png"
            plt.savefig(cum_rewards_png)
            plt.close()
            plt.plot(self.portfolio_return_memory, "r")
            rewards_png=f"{config.RESULTS_DIR}/rewards.png"
            plt.savefig(rewards_png)
            plt.close()

            #print("=================================")
            #print("begin_total_asset: {}".format(self.asset_memory[0]))
            #print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                self.sharpe = (
                    (252 ** 0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )

                #print("Sharpe: ", self.sharpe)
            #print("=================================")

            print(round( self.asset_memory[0],2),round(self.portfolio_value,2),
                         round(df_daily_return["daily_return"].mean(),2),round(df_daily_return["daily_return"].std(),2),
                         round(self.sharpe,2), (df_daily_return["daily_return"].mean()-0.05)/df_daily_return["daily_return"].std() )

            return self.state, self.reward, self.terminal, {}

        else:

            # Normalized actions
            
            if np.array(actions).sum() != 1:
               weights = self.softmax_normalization(actions)
            else: 
               weights = actions

            self.actions_memory.append(weights)

            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].values[0]
            
            self.state = np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            )

            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight

            if  np.mean(self.portfolio_return_memory) > 0.0:
                self.sharpe = (
                        np.mean(self.portfolio_return_memory) / np.std(self.portfolio_return_memory) 
                        )

            if( self.day > 1):
               
                if np.all(self.actions_memory[-1]):
                   txn_cost =  sum( self.transaction_cost_pct * (( self.actions_memory[-2]/self.actions_memory[-1] ) -1) ) 
                else:
                   txn_cost = 0    
                
                portfolio_return = sum( ((self.data.close.values / last_day_memory.close.values) - 1) * weights ) - txn_cost
                                         
            else:
                portfolio_return = sum( ((self.data.close.values / last_day_memory.close.values) - 1) * weights ) - \
                                   sum( self.transaction_cost_pct * weights)
             
             
              
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)

            portfolio_var  = weights.T @ self.covs @ weights
            portfolio_volatity = np.sqrt(portfolio_var)
            
            #print('Day - ',self.day,self.portfolio_value,new_portfolio_value,portfolio_var,portfolio_volatity,portfolio_return)

            self.portfolio_value = new_portfolio_value

            # save into memory

            self.portfolio_return_memory.append(portfolio_return)
 
            self.portfolio_variance.append(portfolio_volatity)

            self.portfolio_account.append( [ w*self.portfolio_value for w in weights ] )

            self.date_memory.append(self.data.date.unique()[0])

            self.asset_memory.append(new_portfolio_value)
            

            # the reward is the new portfolio value or end portfolo value
            #self.reward = portfolio_return

            if portfolio_volatity <= 0.5 and portfolio_return > 0.0:
                self.reward = 100
            else:
                self.reward = -20

            # print("Step reward: ", self.reward)
            # self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # load states
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
                        np.array(self.covs),
                        [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                        axis=0,
                     )

        self.portfolio_value = self.initial_amount

        # self.cost = 0
        # self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

        return self.state

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator      = np.exp(actions)
        denominator    = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def save_portfolio_account(self):
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list,columns=['date'])
        df_date.index = df_date.date
        df_date['variance'] = self.portfolio_variance

        stocks = []
        for s in range(len(self.data.tic.values)):
            stocks.append([])

        for s in self.portfolio_account:
           for index,v in enumerate(s):
               stocks[index].append(v)

        for index,v in enumerate(self.data.tic.values):
            df_date[v] = stocks[index]
        
        account_list = []
        account_list.append([])
        account_list[0]=list(df_date.columns)

        for index,l in enumerate(df_date.columns):
            account_list.append([]) 

        for index,l in enumerate(df_date.columns):
            account_list[index+1].append(list(df_date.iloc[:,index]))

        return account_list


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


