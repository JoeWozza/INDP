# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:46:05 2022

@author: Joe.WozniczkaWells
"""

import pandas as pd
##
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, median_absolute_error
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

basefile = 'INDP/Models/model_reg'

df_VADER = pd.read_csv("INDP//Data//df_VADER_.csv")

# Format tweet_id
df_VADER['tweet_id'] = df_VADER['tweet_id'].astype('Int64').apply(str)

# Categorise sentiment (must be 0, 1 and 2 for to_categorical to work)
#df_VADER['sentiment_cat'] = df_VADER['sentiment_cat']+1

# Use tweets with high or very high sentiment confidence to train model
df_VADER_train = df_VADER[df_VADER['sentconf_cat'].isin(
        ['VeryHigh','High'])].reset_index()
# Create separate test dataset from tweets with low or very low sentiment 
# confidence
df_VADER_test2 = df_VADER[df_VADER['sentconf_cat'].isin(
        ['Low','VeryLow'])].reset_index()

#%%

# Hyperparameter tuning on LSTM model
n_units = [128,256,512]
dropouts = [0,.1,.2]
n_hiddenlayers = [1,2,3]
n_epochs = [25,50,75]
learning_rates = [.001,.01,.1]

# Dataframe containing all combinations of hyperparameters
df_hp1 = pd.DataFrame(list(product(n_units,dropouts,n_hiddenlayers,n_epochs,
                                   learning_rates)),
        columns=['units','dropout','hiddenlayers','epochs','learning_rate'])

df_scores_hp1 = class_lstm.hp_loop(df_VADER_train,df_hp1,basefile,500,
                                   'content_lemma','sentiment',100,
                                   df_VADER_test2,500)

#%% Run a few candidate models on bigger samples

# Chose top 5 based on MSE
df_hp2 = pd.DataFrame([[0.1,75,3,0.001,128],
                       [0.2,50,3,0.01,128],
                       [0.2,75,2,0.001,128],
                       [0.2,75,1,0.001,256],
                       [0.0,75,2,0.001,256]],
                        columns=['dropout','epochs','hiddenlayers',
                                 'learning_rate','units'])

df_scores_hp2 = class_lstm.hp_loop(df_VADER_train,df_hp2,basefile,10000,
                                   'content_lemma','sentiment',100,
                                   df_VADER_test2,10000)

# All perform well, but the best on correlation, mae and mse is 2022-02-18_175759.443774

















