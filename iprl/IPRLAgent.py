# common library
import pandas as pd
import numpy as np
import time
import gym

import iprl.IPRLConfig as config
import iprl.IPRLData as data_split

# RL models from stable-baselines
# from stable_baselines import SAC
# from stable_baselines import TD3

#https://github.com/hill-a/stable-baselines/tree/master/stable_baselines

####SAC from stable_baselines.ppo2 import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines import DDPG
from stable_baselines.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

#from finrl.config import config
#from finrl.preprocessing.data import data_split
#from finrl.env.env_stocktrading import StockTradingEnv

from stable_baselines import A2C
from stable_baselines import PPO2
####SAC from stable_baselines import PPO1,PPO2
from stable_baselines import TD3
###SAC from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from stable_baselines3 import SAC

import tensorflow as tf
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, log_dir: str,verbose=0):
        self.reward = []
        self.mean_reward = []
        self.sharpe = []
        self.log_dir = log_dir
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:

        if self.model.env.envs[0].day==1:
           np_reward      =  np.array(self.reward).cumsum()           
           np_mean_reward =  np.array(self.mean_reward)
           self.reward    = []

           for sharpe in self.sharpe:
               summary = tf.Summary(value=[tf.Summary.Value(tag='Sharpe Ratio', simple_value=sharpe)])
               self.locals['writer'].add_summary(summary,  self.num_timesteps)

#           for sharpe in self.sharpe:
#               summary = tf.Summary(value=[tf.Summary.Value(tag='Sharpe Ratio 2', simple_value=sharpe)])
#               self.locals['writer'].add_summary(summary,  2600)

           for r in self.reward:
               summary = tf.Summary(value=[tf.Summary.Value(tag='Cumulative Reward 2 ', simple_value=r)])
               self.locals['writer'].add_summary(summary,  self.num_timesteps)

#           for r in self.reward:
#               summary = tf.Summary(value=[tf.Summary.Value(tag='Cumulative Reward 4 ', simple_value=r)])
#               self.locals['writer'].add_summary(summary,  2600)


           if ( np_reward.shape[0] > 0 and (np_mean_reward.shape[0] == 0  or
                np_reward.shape[0] == np_mean_reward.shape[0] ) ):

                if np_mean_reward.shape[0] == 0:
                   np_mean_reward = np.zeros(np_reward.shape[0]) 

                np_mean_reward = np.mean([list(np_reward),list(np_mean_reward)],axis=0)

                for r in list(np_mean_reward):
                    summary = tf.Summary(value=[tf.Summary.Value(tag='Mean Cumulative Reward 1', simple_value=r)])
                    self.locals['writer'].add_summary(summary,  self.num_timesteps)

#                for r in list(np_mean_reward):
#                    summary = tf.Summary(value=[tf.Summary.Value(tag='Mean Cumulative Reward 2', simple_value=r)])
#                    self.locals['writer'].add_summary(summary,  2600)

        self.reward.append(self.model.env.envs[0].reward)
        self.sharpe.append(self.model.env.envs[0].sharpe)
        return True
       




######SAC MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC,"ppo": PPO2 }

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class IPRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        train_PPO()
            the implementation for PPO algorithm
        train_A2C()
            the implementation for A2C algorithm
        train_DDPG()
            the implementation for DDPG algorithm
        train_TD3()
            the implementation for TD3 algorithm
        train_SAC()
            the implementation for SAC algorithm
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    @staticmethod
    def DRL_prediction(model, environment):
        test_env, test_obs = environment.get_sb_env()
        """make a prediction"""
        print("Prediction Begins Here : ")
        account_memory = []
        actions_memory = []
        portfolio_account = []
        rewards = []
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs)
            #account_memory = test_env.env_method(method_name="save_asset_memory")
            #actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, reward, dones, info = test_env.step(action)
            rewards.append(reward)
            if i == (len(environment.df.index.unique()) - 2):
              account_memory = test_env.env_method(method_name="save_asset_memory")
              actions_memory = test_env.env_method(method_name="save_action_memory")
              portfolio_account = test_env.env_method(method_name="save_portfolio_account")
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0],rewards,portfolio_account

    def __init__(self, env):
        self.env = env #Monitor(env, f"{config.TENSORBOARD_LOG_DIR}")

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        
        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
        return model

    def train_model(self, model, tb_log_name, total_timesteps=5000):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name,reset_num_timesteps=True, callback=TensorboardCallback(log_dir=f"{config.TENSORBOARD_LOG_DIR}"))
        return model


