import pandas as pd
import iprl.PortfolioAllocEnv as ipenv
import iprl.IPRLAgent as ipagent
import iprl.IPRLConfig as config
import time
from iprl.IPRLData import data_split
from os import path
from random import sample
from stable_baselines3 import A2C

## For Yahoo Finance
import yahoo_fin.stock_info as si
from functools import reduce


def add_cov_matrix(df,lookback=252):
    # add covariance matrix as states
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]
     
    cov_list = []
    for i in range(lookback,len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')        
        return_lookback = price_lookback.pct_change().dropna()
        covs = return_lookback.cov().values 
        cov_list.append(covs)
  
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)

    return df

def train_agent(train,pickle_file,
                agent_type,env_kwargs,parms):
    
    bin_path =  "bin/"+pickle_file

    if(path.exists(bin_path)):
       trained_a2c = A2C.load(bin_path,tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{agent_type}")
    else:          
       e_train_gym = iprl.PortfolioAllocEnv(df = train, **env_kwargs)
       env_train, _ = e_train_gym.get_sb_env()      

       agent = ipagent.IPRLAgent(env = env_train)

       model_a2c = agent.get_model(model_name=agent_type,model_kwargs = parms)

       trained_a2c = agent.train_model(model=model_a2c, 
                           tb_log_name=agent_type,
                           total_timesteps=60000)

       trained_a2c.save(bin_path) 

    return trained_a2c

def get_a2c_kwargs(state_space,stock_dimension):

    env_kwargs = {
               "hmax": 100, 
               "initial_amount": 10000, 
               "transaction_cost_pct": 0.001, 
               "state_space": state_space, 
               "stock_dim": stock_dimension, 
               "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
               "action_space": stock_dimension, 
               "reward_scaling": 1e-4 }

    return env_kwargs


def a2c_model(train_data,env_kwargs):

    A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}

    model = train_agent(train_data, "a2c_agent.p", "a2c", env_kwargs, A2C_PARAMS)

    return model


def main():
    sentiments = [0,1,2]
    
    df = pd.read_csv('data/dow30etf_20210523-124004.csv')    

    df['sentiments'] = [sample(sentiments,1)[0] for i in range(df.shape[0]) ]
    
    df = add_cov_matrix(df)

    train_data = data_split(df, '2009-01-01','2019-01-01')

    stock_dimension = len(train_data.tic.unique())
    
    state_space = stock_dimension

    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = get_a2c_kwargs(state_space,stock_dimension)

    a2c_agent = a2c_model(train_data,env_kwargs)

    trade_data = data_split(df,'2019-01-01', '2021-01-01')

    e_trade_gym = ipenv.PortfolioAllocEnv(trade_data, **env_kwargs)            
    
    df_daily_return, df_actions = ipagent.IPRLAgent.DRL_prediction(model=a2c_agent, environment = e_trade_gym)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    actions_filename = 'agent_actions_'+ timestamp+'.csv'

    df_actions.to_csv(actions_filename)
     
    daily_returns_filename = 'daily_return_'+timestamp+'.csv'
    
    df_daily_return.to_csv(daily_returns_filename)

if __name__ == "__main__":
    main()




