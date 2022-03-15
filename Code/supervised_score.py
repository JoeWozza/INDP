# -*- coding: utf-8 -*-
"""
This script scores the final model on the TweePy and sntwitter data.

@author: Joe Wozniczka-Wells
"""

import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.ticker as tkr
from datetime import datetime
import pickle
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

#%% Score on TweePy Midlands data
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
X_text_tweepy = df_tweets_deduped['tweet_text'].apply(class_v.clean)
# LSTM cleaning: lemmatisation and tokenisation
X_tweepy = class_lstm.score_prep(X_text_tweepy,tokenizer,100)
# Score model on Midlands tweets
df_tweets_deduped['LSTM_sent'] = model.predict(X_tweepy)
df_tweets_deduped['tweet_text_clean'] = X_text_tweepy

df_tweets_deduped.to_csv("{0}/df_tweepy_LSTM_sent.csv".format(lstm_folder))

#%% Score on sntwitter Midlands data

# Define common string at start of file names
filestring_sntwitter = 'df_tweets_deduped_sntwitter_'
# Get list of all files containing common string
tweets_files_sntwitter = ['INDP/Data/Tweets/{0}'.format(s) for s in files if 
                          filestring_sntwitter in s]
# Select most recent file
max_file = max(tweets_files_sntwitter, key=os.path.getctime)
# Read in file
df_tweets_sntwitter = pd.read_csv(max_file)

# VADER cleaning: remove twitter handles, URLs and most special characters
X_text_sntwitter = df_tweets_sntwitter['tweet_text'].apply(class_v.clean)
# LSTM cleaning: lemmatisation and tokenisation
X_sntwitter = class_lstm.score_prep(X_text_sntwitter,tokenizer,100)
# Score model on Midlands tweets
df_tweets_sntwitter['LSTM_sent'] = model.predict(X_sntwitter)
df_tweets_sntwitter['tweet_text_clean'] = X_text_sntwitter

df_tweets_sntwitter.to_csv("{0}/df_sntwitter_LSTM_sent.csv".format(
        lstm_folder))
