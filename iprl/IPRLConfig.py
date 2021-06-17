import pathlib
import pandas as pd
import datetime
import os

# pd.options.display.max_rows = 10
# pd.options.display.max_columns = 10


# PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
# PACKAGE_ROOT = pathlib.Path().resolve().parent

TRAINED_MODEL_DIR = f"trained_models"
# DATASET_DIR = PACKAGE_ROOT / "data"

# data
# TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
# TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
# TESTING_DATA_FILE = "test.csv"

# now = datetime.datetime.now()
# TRAINED_MODEL_DIR = f"trained_models/{now}"

DATA_SAVE_DIR = f"data"
TRAINED_MODEL_DIR = f"trained_models"
TENSORBOARD_LOG_DIR = f"tensorboard_log"
RESULTS_DIR = f"results"

# os.makedirs(TRAINED_MODEL_DIR)


## time_fmt = '%Y-%m-%d'

START_DATE = "2000-01-01"
END_DATE = "2021-01-01"

START_TRADE_DATE = "2019-01-01"

## dataset default columns
DEFAULT_DATA_COLUMNS = ["date", "tic", "close"]

##
## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
##

TECHNICAL_INDICATORS_LIST = ["macd","boll_ub","boll_lb","rsi_30", "cci_30", "dx_30","close_30_sma","close_60_sma"]

##
## Model Parameters
##

A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.07}

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}

TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}

SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 64,
    "ent_coef": "auto_0.1",
}

