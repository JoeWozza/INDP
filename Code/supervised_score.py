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
from tensorflow.keras.models import load_model
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
from itertools import product

from os import chdir, listdir
import os

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

from INDP.Code import LSTM
class_lstm = LSTM.LSTM()
from INDP.Code import VADER
class_v = VADER.VADER()

# Create folders
hp2_folder = 'INDP/Models/hp2'
lstm_folder = 'INDP/Data/LSTM'

if not os.path.exists(lstm_folder):
    os.makedirs(lstm_folder)

#%% Read in all Midlands data
# Get list of all files in path
files = listdir("INDP/Data/Tweets")

# Define common string at start of file names
filestring_utlas = 'df_tweets_tweepy_'
# Get list of all files containing common string
tweets_files_utlas = [s for s in files if filestring_utlas in s]
# Combine data into single dataframe
df_tweets = pd.DataFrame()
for f in tweets_files_utlas:
    print(f)
    df_tweets = df_tweets.append(pd.read_csv("INDP/Data/Tweets/{0}".format(f)))

# Deduplicate by tweet_id and UTLA
df_tweets_deduped = (df_tweets
                     .drop_duplicates(subset=['tweet_id','area'])
                     .reset_index().drop(columns=['index','Unnamed: 0']))

# Load model
model_timestamp = '2022-02-18_175759.443774'

model = load_model('{0}/{1}/model_reg.h5'.format(hp2_folder,model_timestamp))

# Load tokenizer associated with model
with open('{0}/{1}/tokenizer.pickle'.format(hp2_folder,model_timestamp), 
          'rb') as handle:
    tokenizer = pickle.load(handle)

# VADER cleaning: remove twitter handles, URLs and most special characters
X_text = df_tweets_deduped['tweet_text'].apply(class_v.clean)
# LSTM cleaning: lemmatisation and tokenisation
X = class_lstm.score_prep(X_text,tokenizer,100)
# Score model on Midlands tweets
df_tweets_deduped['LSTM_sent'] = model.predict(X)
df_tweets_deduped['tweet_text_clean'] = X_text

df_tweets_deduped.to_csv("{0}/df_LSTM_sent.csv".format(lstm_folder))



