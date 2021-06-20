import pandas as pd
import numpy as np
import time

import iprl.PortfolioAllocEnv as ipenv
import iprl.IPRLAgent as ipagent
import iprl.IPRLConfig as config
from iprl.IPRLData import data_split
from tqdm import tqdm

from random import sample

from stable_baselines import A2C
from stable_baselines.ddpg import DDPG
from stable_baselines.ppo2 import PPO2
from torch.utils.tensorboard import SummaryWriter

from os import path

## For Yahoo Finance
import yahoo_fin.stock_info as si
from functools import reduce

def add_cov_matrix(df,lookback=252):
    # add covariance matrix as states
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]
    
    cov_list = []
    for i in tqdm(range(lookback,len(df.index.unique()))):
        data_lookback = df.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close') 
        return_lookback = price_lookback.pct_change().dropna()#.apply(lambda x: np.log(1+x))
        covs = return_lookback.cov().values 
        cov_list.append(covs)
    
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)

    return df

def add_cov_matrix_new(df):

    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]
    adjclose_df =df.pivot_table(index = 'date',columns = 'tic', values = 'close').dropna() 
    returns_df = adjclose_df.pct_change().dropna()

    cov_list = []

    for i in tqdm(range(2,len(returns_df.index.unique()))):
        covs = returns_df.iloc[:i,:].cov().values*252
        cov_list.append(covs)

    df_cov = pd.DataFrame({'date':adjclose_df.reset_index()['date'].values[:-3],'cov_list':cov_list})
    final_df = df.merge(df_cov, on='date')
    final_df = final_df.sort_values(['date','tic']).reset_index(drop=True)

    return final_df



def train_agent(train,pickle_file,
                agent_type,env_kwargs,parms):
    
    bin_path =  "bin/"+pickle_file

    if (path.exists(bin_path)):
         if agent_type == "a2c":
            print("Loading A2C Agent")
            RL_model = A2C.load(bin_path,tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{agent_type}")
         elif agent_type == "ddpg":           
            print("Loading DDPG Agent")
            RL_model = DDPG.load(bin_path,tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{agent_type}")
         elif agent_type == "ppo":
            print("Loading PPO2 Agent") 
            RL_model = PPO2.load(bin_path,tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{agent_type}")
    else:
       e_train_gym = ipenv.PortfolioAllocEnv(df = train, **env_kwargs)
       env_train, _ = e_train_gym.get_sb_env()      

       agent = ipagent.IPRLAgent(env = env_train)

       model = agent.get_model(model_name=agent_type,model_kwargs = parms)

       RL_model = agent.train_model(model=model,
                           tb_log_name=agent_type,
                           total_timesteps=1000000)

       RL_model.save(bin_path) 

    return RL_model

def merge_with_dji(df,outfile_name):
    
    filename = 'data/dji_20210620-133812.csv'

    print('Filename  ',filename)

    dji_df  = pd.read_csv(filename)
    
    trade_dji = data_split(dji_df,'2019-01-01', '2021-06-11')  
    
    trade_dji=trade_dji.sort_values(['date','tic'],ignore_index=True)
    trade_dji.index = trade_dji.date.factorize()[0]
    
    trade_close=trade_dji.pivot_table(index = 'date',columns = 'tic', values = 'close') 
    
    trade_dji_daily = trade_close.pct_change().fillna(0)#.apply(lambda x: np.log(1+x))

    trade_dji_daily = df.merge(trade_dji_daily, on='date')
    trade_dji_daily = trade_dji_daily.sort_values(['date']).reset_index(drop=True)
    
    trade_dji_daily.to_excel(outfile_name)
    
    return df



def a2c_training(train_data,trade_data,state_space,stock_dimension):

    writer = SummaryWriter(log_dir=f"{config.TENSORBOARD_LOG_DIR}")

    env_kwargs = {
               "hmax": 100, 
               "initial_amount": 1000000, 
               "transaction_cost_pct": 0.001, 
               "state_space": state_space, 
               "stock_dim": stock_dimension, 
               "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
               "action_space": stock_dimension, 
               "reward_scaling": 1e-4 }

    A2C_PARAMS = {"n_steps": 2, "ent_coef": 0.05,  "learning_rate": 0.005}

    a2c_agent = train_agent(train_data, "a2c_agent.p", "a2c", env_kwargs, A2C_PARAMS)

    e_trade_gym = ipenv.PortfolioAllocEnv(trade_data, **env_kwargs)
    
    df_daily_return, df_actions,rewards,p_acct = ipagent.IPRLAgent.DRL_prediction(model=a2c_agent, environment = e_trade_gym)
    
    for index,r in enumerate(np.array(rewards).cumsum()):
        writer.add_scalar('Trading/a2creward', r, index+1)

 
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    actions_filename = 'reports/a2c_agent_actions_'+ timestamp+'.csv'

    df_actions.to_csv(actions_filename)
     
    daily_returns_filename = 'reports/a2c_daily_return_'+timestamp+'.xlsx'
    
    merge_with_dji(df_daily_return,daily_returns_filename)
    
    portfolio_account_filename = 'reports/a2c_account_'+timestamp+'.csv'

    df_acct = pd.DataFrame(columns=p_acct[0][0])
    
    for index,lst in enumerate(p_acct[0]):
        if index > 0:
           df_acct[ df_acct.columns.to_list()[index-1] ] = lst[0]
    
    df_acct.to_csv(portfolio_account_filename)
    
    


    #rewards = 'a2c_cum_rewards_'+timestamp+'.csv'
    
    #rewards.to_csv(rewards.cumsum())


def ppo_training(train_data,trade_data,state_space,stock_dimension):    

    writer = SummaryWriter(log_dir=f"{config.TENSORBOARD_LOG_DIR}")

    env_kwargs = {
               "hmax": 100, 
               "initial_amount": 1000000, 
               "transaction_cost_pct": 0.001, 
               "state_space": state_space, 
               "stock_dim": stock_dimension, 
               "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
               "action_space": stock_dimension, 
               "reward_scaling": 1e-4 }

    PPO_PARAMS = {
               "n_steps": 128,
               "ent_coef": 0.05,
               "learning_rate": 0.005}

    ppo_agent = train_agent(train_data, "ppo_agent.p", "ppo", env_kwargs, PPO_PARAMS)


    e_trade_gym = ipenv.PortfolioAllocEnv(trade_data, **env_kwargs)

    df_daily_return, df_actions, rewards, p_acct = ipagent.IPRLAgent.DRL_prediction(model=ppo_agent, environment = e_trade_gym)

    for index,r in enumerate(np.array(rewards).cumsum()):
        writer.add_scalar('Trading/pporeward', r, index+1)


    timestamp = time.strftime("%Y%m%d-%H%M%S")

    actions_filename = 'reports/ppo_agent_actions_'+ timestamp+'.csv'

    df_actions.to_csv(actions_filename)
     
    daily_returns_filename = 'reports/ppo_daily_return_'+timestamp+'.xlsx'

    merge_with_dji(df_daily_return,daily_returns_filename)
    
    portfolio_account_filename = 'reports/ppo_account_'+timestamp+'.csv'

    df_acct = pd.DataFrame(columns=p_acct[0][0])
    
    for index,lst in enumerate(p_acct[0]):
        if index > 0:
           df_acct[ df_acct.columns.to_list()[index-1] ] = lst[0]
    
    df_acct.to_csv(portfolio_account_filename)


    #rewards = 'ppo_cum_rewards_'+timestamp+'.csv'
    
    #rewards.to_csv(rewards.cumsum())

def ddpg_training(train_data,trade_data,state_space,stock_dimension):

    writer = SummaryWriter(log_dir=f"{config.TENSORBOARD_LOG_DIR}")

    env_kwargs = {
               "hmax": 100, 
               "initial_amount": 1000000, 
               "transaction_cost_pct": 0.001, 
               "state_space": state_space, 
               "stock_dim": stock_dimension, 
               "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
               "action_space": stock_dimension, 
               "reward_scaling": 1e-4 }
    
    DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000}

    ddpg_agent = train_agent(train_data, "ddpg_agent.p", "ddpg", env_kwargs, DDPG_PARAMS)

    e_trade_gym = ipenv.PortfolioAllocEnv(trade_data, **env_kwargs)
    
    df_daily_return, df_actions, rewards,p_acct = ipagent.IPRLAgent.DRL_prediction(model=ddpg_agent, environment = e_trade_gym)

    for index,r in enumerate(np.array(rewards).cumsum()):
        writer.add_scalar('Trading/ddpgreward', r, index+1)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    actions_filename = 'reports/ddpg_agent_actions_'+ timestamp+'.csv'

    df_actions.to_csv(actions_filename)
     
    daily_returns_filename = 'reports/ddpg_daily_return_'+timestamp+'.xlsx'
    
    merge_with_dji(df_daily_return,daily_returns_filename)

    portfolio_account_filename = 'reports/ddpg_account_'+timestamp+'.csv'

    df_acct = pd.DataFrame(columns=p_acct[0][0])
    
    for index,lst in enumerate(p_acct[0]):
        if index > 0:
           df_acct[ df_acct.columns.to_list()[index-1] ] = lst[0]
    
    df_acct.to_csv(portfolio_account_filename)


    #rewards = 'ddpg_cum_rewards_'+timestamp+'.csv'
    
    #rewards.to_csv(rewards.cumsum())


def main():

    #df = pd.read_csv('data/dow30etf_20210523-124004.csv')
    #df = pd.read_csv('data/dow30etf_20210606-153730.csv')

    #df  = pd.read_csv('data/dow30_20210613-180320.csv')
    filename = 'data/dow30_20210620-145037.csv'
    print('Filename  ',filename)

    df  = pd.read_csv(filename)

 ##   df_full = data_split(df,'2009-01-01','2021-06-11')
 ##   df_cov  = add_cov_matrix(df_full)

 ##    train_data = data_split(df_cov,'2009-01-01','2019-01-01')
 ##   trade_data = data_split(df_cov,'2019-01-01', '2021-06-11')

    df_cov  = add_cov_matrix(df)
    train_data = data_split(df_cov,'2009-01-01','2018-01-01')
    trade_data = data_split(df_cov,'2019-01-01', '2021-06-11')


    stock_dimension = len(train_data.tic.unique())

    state_space = stock_dimension

    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    #print('A2C Training Started  : ',time.strftime("%Y%m%d-%H%M%S"))
    #a2c_training(train_data,trade_data,state_space,stock_dimension)
    #print('A2C Training Finished : ',time.strftime("%Y%m%d-%H%M%S"))

    #print('DDPG Training Started  : ',time.strftime("%Y%m%d-%H%M%S"))
    #ddpg_training(train_data,trade_data,state_space,stock_dimension)
    #print('DDPG Training Finished : ',time.strftime("%Y%m%d-%H%M%S"))

    print('PPO Training Started  : ',time.strftime("%Y%m%d-%H%M%S"))
    ppo_training(train_data,trade_data,state_space,stock_dimension)
    print('PPO Training Finished : ',time.strftime("%Y%m%d-%H%M%S"))

    #timestamp = time.strftime("%Y%m%d-%H%M%S")
    #train_filename = 'data/train_data_'+timestamp+'.csv'
    #df_full = add_cov_matrix(df_full)
    #df_full.iloc[:,:-1].to_csv(train_filename)


if __name__ == "__main__":
    main()

#df  = pd.read_csv('/Users/sachin/Downloads/done_data-4.csv')
#df.info()
#df.groupby(['tic'])['date'].min().sort_values()


#df.groupby(['tic'])['date'].min()

df = pd.read_excel('reports/ppo_daily_return_20210620-161117.xlsx')



