import json
import logging
import os
import time
from argparse import ArgumentParser
import datetime
import pandas as pd

import iprl.IPRLConfig as config
from iprl.IPRLData import data_split

## For Yahoo Finance
import yahoo_fin.stock_info as si
from functools import reduce


from finrl.preprocessing.preprocessors import FeatureEngineer

from functools import reduce

def build_parser():

    parser = ArgumentParser()

    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    return parser

def initialize():
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
       os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
       os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
       os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
       os.makedirs("./" + config.RESULTS_DIR)

def main():
    initialize()
    parser = build_parser()
    options = parser.parse_args()

    if options.mode == "train":
       import finrl.autotrain.training

       finrl.autotrain.training.train_one()
    elif options.mode == "download_data":
        print('Download Data Begin')

        dow_30 = si.tickers_dow()
        # ETF
        #dftmp = pd.read_csv('data/etf_tom.csv',index_col=0)
        #dow_30 = dftmp.tic.unique()

        # DOW30

        dftmp = pd.read_csv('data/tom_dow_done_data.csv',index_col=0)
        dow_30 = dftmp.tic.unique()
        dow_30 = ['DSS','AAPL','INFY']
        price_data = {ticker : si.get_data(ticker) for ticker in dow_30}
        df = reduce(lambda x,y: x.append(y), price_data.values())
        df.reset_index(inplace=True)
        df = df.rename(columns = {'index':'date','ticker':'tic'})

        fe = FeatureEngineer(
                        use_technical_indicator=True,
                        use_turbulence=True,
                        user_defined_feature = False)
        
        df = fe.preprocess_data(df)
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        df.to_csv(config.DATA_SAVE_DIR + "/" + "dow30_"+ now + ".csv",index=False)
        print('Download Complete')

if __name__ == "__main__":
    main()


 