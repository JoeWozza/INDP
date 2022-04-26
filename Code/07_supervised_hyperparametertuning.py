# -*- coding: utf-8 -*-
"""
This code is used to perform two rounds of hyperparameter tuning on the LSTM
model and select the best model.

@author: Joe Wozniczka-Wells
"""

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, median_absolute_error
import seaborn as sns
import matplotlib.ticker as tkr
from datetime import datetime
import pickle
import numpy as np
import os
from itertools import product
from os import chdir, listdir

# Set file path
filepath = ("C:\\Users\\joew\\Documents\\Apprenticeship\\UoB\\SPFINDP21T4\\")
chdir(filepath)

# Create folder in which to save visualisations
models_folder = 'INDP/Models'
hp1_folder = 'INDP/Models/hp1'
hp2_folder = 'INDP/Models/hp2'

if not os.path.exists(models_folder):
    os.makedirs(models_folder)
if not os.path.exists(hp1_folder):
    os.makedirs(hp1_folder)
if not os.path.exists(hp2_folder):
    os.makedirs(hp2_folder)

from INDP.Code import LSTM
class_lstm = LSTM.LSTM()
from INDP.Code import VADER
class_v = VADER.VADER()

df_VADER = pd.read_csv("INDP/Data/VADER/df_VADER.csv")

df_VADER['tweet_text_clean'] = df_VADER['tweet_text'].apply(class_v.clean)

# Format tweet_id
df_VADER['tweet_id'] = df_VADER['tweet_id'].astype('Int64').apply(str)

# Categorise sentiment (must be 0, 1 and 2 for to_categorical to work)
#df_VADER['sentiment_cat'] = df_VADER['sentiment_cat']+1

# Categorise sentiment confidence score
mean_cs,std_cs = class_v.cat_sentconf_stats(df_VADER['sentconf'])
df_VADER['sentconf_cat'] = df_VADER.apply(lambda row:
    class_v.cat_sentconf(row['sentconf'],mean_cs,std_cs), axis=1)

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

df_scores_hp1 = class_lstm.hp_loop(df_VADER_train,df_hp1,hp1_folder,500,
                                   'tweet_text_clean','sentiment',100,
                                   df_VADER_test2,500)
df_scores_hp1.to_csv('{0}/df_scores_hp1.csv'.format(hp1_folder))

# Look into average performance by each value for each hyperparameter
df_scores_hp1_melt = df_scores_hp1.melt(id_vars=['model','test_mae','test_mse',
                                                 'test_corr','test_auc'],
                                        value_vars=['units','dropout',
                                                    'hiddenlayers','epochs',
                                                    'learning_rate'],
                                        value_name='hp_value',
                                        var_name='hp_name')
df_scores_hp1_melt['hp_value'] = df_scores_hp1_melt['hp_value'].astype(str)

df_scores_hp1_melt2 = df_scores_hp1_melt.melt(id_vars=['model','hp_name',
                                                       'hp_value'],
                                        value_vars=['test_mae','test_mse',
                                                    'test_corr','test_auc'],
                                        var_name='metric')

# Plot distribution of metrics by hyperparameters
g = sns.FacetGrid(df_scores_hp1_melt2,row='hp_name',col='metric',sharex=False)
g.map(sns.boxplot, 'hp_value', 'value', color='#007C91', showfliers=False)
g.set(ylim=(-1,1))
g.set_axis_labels(x_var = 'Hyperparameter value', y_var = 'Metric value')
g.set_titles(col_template = '{col_name}', row_template = '{row_name}')
g.tight_layout()
g.savefig("{0}/hp1_testmetrics.png".format(hp1_folder))

#%% Run a few candidate models on bigger samples

# Chose top 5 based on MSE
df_hp2 = pd.DataFrame([[0.1,75,3,0.001,128],
                       [0.2,50,3,0.01,128],
                       [0.2,75,2,0.001,128],
                       [0.2,75,1,0.001,256],
                       [0.0,75,2,0.001,256]],
                        columns=['dropout','epochs','hiddenlayers',
                                 'learning_rate','units'])

df_scores_hp2 = class_lstm.hp_loop(df_VADER_train,df_hp2,hp2_folder,10000,
                                   'tweet_text_clean','sentiment',100,
                                   df_VADER_test2,10000)
df_scores_hp2.to_csv('{0}/df_scores_hp2.csv'.format(hp2_folder))
# All perform well, but the best on correlation, mae and mse is 
# 2022-02-18_175759.443774: 0.2, 50.0, 3.0, 0.01, 128.0
