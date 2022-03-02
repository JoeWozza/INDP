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


#%% Ad-hocL calculate AUC on all hp1 models

## Get results from all hyperparameter combinations into dataframe
# Define common string in folder name
folderstring = '2022-'
# Get list of model timestamps
all_folders = listdir(hp1_folder)
folders = [s for s in all_folders if folderstring in s]

# Remove these three (from hp2): 2022-02-18_213521.783721, 2022-02-18_175759.443774, 2022-02-18_135831.330432
#folders.remove('2022-02-18_213521.783721')
#folders.remove('2022-02-18_175759.443774')
#folders.remove('2022-02-18_135831.330432')

# Loop through folders and append results to dataframe
df_scores = pd.DataFrame()
for f in folders:
    print(f)
    score_dict = np.load('{0}/{1}/score_dict.npy'.format(hp1_folder,f),allow_pickle='TRUE').item()
    score_dict['model'] = f
    df_scores = df_scores.append(score_dict, ignore_index=True)

from tensorflow.keras.models import load_model

# Recreate old score_dicts
for index,row in df_scores.iterrows():
    score_dict = {'model': row['model'],
                  'units': row['units'],
                  'dropout': row['dropout'],
                  'hiddenlayers': row['hiddenlayers'],
                  'epochs': row['epochs'],
                  'learning_rate': row['learning_rate'],
                      'time_taken': row['time_taken'],
                      'test_mse': row['test_mse'],
                      'test_mae': row['test_mae'],
                      'test_corr': row['test_corr'],
                      'test2_mse': row['test2_mse'],
                      'test2_mae': row['test2_mae'],
                      'test2_corr': row['test2_corr']
                          }
    np.save('{0}/{1}/score_dict'.format(hp1_folder,row['model']),score_dict)

# Add AUC to existing score_dicts
df_train = df_VADER_train.sample(n=500)
X_test = df_train['content_lemma']
y_test = df_train['sentiment']
df_test2 = df_VADER_test2.sample(n=500)
X_test2 = df_test2['content_lemma']
y_test2 = df_test2['sentiment']
for index,row in df_scores.iterrows():
    
    units = row['units']
    dropout = row['dropout']
    hiddenlayers = row['hiddenlayers']
    epochs = row['epochs']
    learning_rate = row['learning_rate']
    start = datetime.now()
    end = datetime.now()
    model_timestamp = row['model']
    print(model_timestamp)
    
    # Load tokenizer and model
    with open('{0}/{1}/tokenizer.pickle'.format(hp1_folder,model_timestamp), 
              'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model('{0}/{1}/model_reg.h5'.format(hp1_folder,model_timestamp))
    # Apply tokenizer
    vocab_size=len(tokenizer.word_index)+1
    X_test_=pad_sequences(tokenizer.texts_to_sequences(X_test),maxlen=100)
    X_test2_=pad_sequences(tokenizer.texts_to_sequences(X_test2),maxlen=100)
    
    # Score model on test dataset
    predict_test = model.predict(X_test_)
    # Score model on 'test2' dataset                
    predict_test2 = model.predict(X_test2_)
    
    score_dict = np.load('{0}/{1}/score_dict.npy'.format(hp1_folder,model_timestamp),allow_pickle='TRUE').item()
    
    score_dict = {'model': score_dict['model'],
                  'units': score_dict['units'],
                  'dropout': score_dict['dropout'],
                  'hiddenlayers': score_dict['hiddenlayers'],
                  'epochs': score_dict['epochs'],
                  'learning_rate': score_dict['learning_rate'],
                  'time_taken': score_dict['time_taken'],
                  'test_mse': score_dict['test_mse'],
                  'test_mae': score_dict['test_mae'],
                  'test_auc':class_lstm.auc_score(y_test,predict_test),
                  'test_corr': score_dict['test_corr'],
                  'test2_mse': score_dict['test2_mse'],
                  'test2_mae': score_dict['test2_mae'],
                  'test2_auc':class_lstm.auc_score(y_test2,predict_test2),
                  'test2_corr': score_dict['test2_corr']
                          }
    np.save('{0}/{1}/score_dict_'.format(hp1_folder,model_timestamp),
            score_dict)

2022-02-11_080751.667393

score_dict = np.load('{0}/2022-02-09_164246.262937/score_dict.npy'.format(hp1_folder),allow_pickle='TRUE').item()

df_scores_ = pd.DataFrame()
for f in folders:
    print(f)
    score_dict = np.load('{0}/{1}/score_dict_.npy'.format(hp1_folder,f),allow_pickle='TRUE').item()
    score_dict['model'] = f
    df_scores_ = df_scores_.append(score_dict, ignore_index=True)

# Check the MAE and MSE values of the top 5 are the same as in df_scores and hope the AUCs for these models aren't bad
# They are the same. Most of the AUCs are okay, one is only 0.61.

# Add AUC to existing score_dicts - hp2
df_train = df_VADER_train.sample(n=1000)
X_test = df_train['content_lemma']
y_test = df_train['sentiment']
df_test2 = df_VADER_test2.sample(n=10000)
X_test2 = df_test2['content_lemma']
y_test2 = df_test2['sentiment']
for index,row in df_scores_hp2.iterrows():
    model_timestamp = row['model']
    print(model_timestamp)
    
    # Load tokenizer and model
    with open('{0}/{1}/tokenizer.pickle'.format(hp2_folder,model_timestamp), 
              'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model('{0}/{1}/model_reg.h5'.format(hp2_folder,model_timestamp))
    # Apply tokenizer
    vocab_size=len(tokenizer.word_index)+1
    X_test_=pad_sequences(tokenizer.texts_to_sequences(X_test),maxlen=100)
    X_test2_=pad_sequences(tokenizer.texts_to_sequences(X_test2),maxlen=100)
    
    # Score model on test dataset
    predict_test = model.predict(X_test_)
    # Score model on 'test2' dataset                
    predict_test2 = model.predict(X_test2_)
    
    score_dict = np.load('{0}/{1}/score_dict.npy'.format(hp2_folder,model_timestamp),allow_pickle='TRUE').item()
    
    score_dict = {'model': score_dict['model'],
                  'units': score_dict['units'],
                  'dropout': score_dict['dropout'],
                  'hiddenlayers': score_dict['hiddenlayers'],
                  'epochs': score_dict['epochs'],
                  'learning_rate': score_dict['learning_rate'],
                  'time_taken': score_dict['time_taken'],
                  'test_mse': score_dict['test_mse'],
                  'test_mae': score_dict['test_mae'],
                  'test_auc':class_lstm.auc_score(y_test,predict_test),
                  'test_corr': score_dict['test_corr'],
                  'test2_mse': score_dict['test2_mse'],
                  'test2_mae': score_dict['test2_mae'],
                  'test2_auc':class_lstm.auc_score(y_test2,predict_test2),
                  'test2_corr': score_dict['test2_corr']
                          }
    np.save('{0}/{1}/score_dict_'.format(hp2_folder,model_timestamp),
            score_dict)

df_scores_hp2_ = pd.DataFrame()
for index,row in df_scores_hp2.iterrows():
    f = row['model']
    score_dict = np.load('{0}/{1}/score_dict_.npy'.format(hp2_folder,f),allow_pickle='TRUE').item()
    score_dict['model'] = f
    df_scores_hp2_ = df_scores_hp2_.append(score_dict, ignore_index=True)


