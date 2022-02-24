# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 08:54:07 2022

@author: Joe.WozniczkaWells
"""

import pandas as pd
##
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
#from sklearn.model_selection import train_test_split
#from tensorflow.keras.models import Sequential
#from tensorflow.keras import layers
#from tensorflow.keras.optimizers import Adam
#from sklearn.metrics import mean_squared_error, median_absolute_error
#import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tkr
from datetime import datetime
import pickle
##
import numpy as np
import os
from itertools import product

from os import chdir, listdir

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

from INDP.Code import LSTM
class_lstm = LSTM.LSTM()
from INDP.Code import TweetScrape
class_ts = TweetScrape.TweetScrape()

basefile = 'INDP/Models/model_reg'

#%% Read in all Midlands data
# Get list of all files in path
files = listdir("{0}\\INDP\\Data".format(filepath))

# Define common string at start of file names
filestring_mids = 'df_mids_tweets_tweepy_'
filestring_utlas = 'df_tweets_tweepy_'
# Get list of all files containing common string
tweets_files_mids = [s for s in files if filestring_mids in s]
tweets_files_utlas = [s for s in files if filestring_utlas in s]
# Combine data into single dataframes
df_tweets_mids = pd.DataFrame()
for f in tweets_files_mids:
    print(f)
    df_tweets_mids = df_tweets_mids.append(pd.read_csv("INDP\\Data\\" + f))
df_tweets_utlas = pd.DataFrame()
for f in tweets_files_utlas:
    print(f)
    df_tweets_utlas = df_tweets_utlas.append(pd.read_csv("INDP\\Data\\" + f))

# Deduplicate by tweet_id
df_tweets_utlas_deduped = df_tweets_utlas.drop_duplicates(
        subset=['tweet_id','area']).reset_index().drop(columns=
               ['index','Unnamed: 0'])

# Check whether any of Midlands tweets (that haven't already) can be assigned 
# to UTLAs
df_utlas = pd.read_csv("{0}/INDP/Data/df_utlas_90.csv".format(filepath))
utlas = df_utlas.drop_duplicates(subset=['utla']).utla

df_tweets_utlas_deduped_plus = df_tweets_utlas_deduped.append(
        class_ts.manual_assign_areas(df_tweets_mids,utlas)).drop_duplicates(
        subset=['tweet_id','area'])

print(len(df_tweets_utlas_deduped))
print(len(df_tweets_utlas_deduped_plus))


