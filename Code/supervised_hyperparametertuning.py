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

from os import chdir, listdir

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

basefile = 'INDP/Models/model_reg'

df_VADER = pd.read_csv("INDP//Data//df_VADER.csv")

# Format tweet_id
df_VADER['tweet_id'] = df_VADER['tweet_id'].astype('Int64').apply(str)

# Categorise sentiment (must be 0, 1 and 2 for to_categorical to work)
def cat_sentiment(row):
    # Positive
    if row['sentiment'] >= 0.05:
        return 2
    # Negative
    elif row['sentiment'] <= - 0.05:
        return 0
    # Neutral
    else:
        return 1

def cat_setconf(row):
    if row['sentconf'] >= thr:
        return 'VeryHigh'
    elif row['sentconf'] >= thr - 0.5 * std:
        return 'High'
    elif row['sentconf'] >= thr - std:
        return 'Low'
    elif row['sentconf'] > 0:
        return 'VeryLow'
    else:
        return 'Zero'

# Code taken from https://towardsdatascience.com/sentiment-analysis-comparing-3-common-approaches-naive-bayes-lstm-and-vader-ab561f834f89

def data_cleaning(text_list): 
    stopwords_rem=False
    stopwords_en=stopwords.words('english')
    lemmatizer=WordNetLemmatizer()
    tokenizer=TweetTokenizer()
    reconstructed_list=[]
    for each_text in text_list:
        lemmatized_tokens=[]
        tokens=tokenizer.tokenize(each_text.lower())
        pos_tags=pos_tag(tokens)
        for each_token, tag in pos_tags: 
            if tag.startswith('NN'): 
                pos='n'
            elif tag.startswith('VB'): 
                pos='v'
            else: 
                pos='a'
            lemmatized_token=lemmatizer.lemmatize(each_token, pos)
            if stopwords_rem: # False 
                if lemmatized_token not in stopwords_en: 
                    lemmatized_tokens.append(lemmatized_token)
            else: 
                lemmatized_tokens.append(lemmatized_token)
        reconstructed_list.append(' '.join(lemmatized_tokens))
    return reconstructed_list

def get_dataset(row):
    if "_" in row['variable']:
        val = 'valid'
    else:
        val = 'train'
    return val

df_VADER['sentiment_cat'] = df_VADER.apply(lambda row: cat_sentiment(row), axis=1)

# Categorise sentiment confidence
mcs = df_VADER.sentconf.mean()
std = df_VADER.sentconf.std()
thr = mcs + 0.5 * std

df_VADER['sentconf_cat'] = df_VADER.apply(lambda row: cat_setconf(row), axis=1)

df_VADER_train = df_VADER[df_VADER['sentconf'] >= thr - 0.5 * std].reset_index()
df_VADER_test2 = df_VADER[(df_VADER['sentconf'] < thr - 0.5 * std) & 
                         (df_VADER['sentconf'] > 0)].reset_index()

#%%
df_VADER_train_samp = df_VADER_train.sample(n=500)
df_VADER_test2_samp = df_VADER_test2.sample(n=500)
X=df_VADER_train_samp['content_lemma']
y=df_VADER_train_samp[['sentiment_cat','sentiment']]
#X=X.head(1000)
#y=y.head(1000)
X_train, X_testval, y_train, y_testval=train_test_split(X, y, test_size=.3)
X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, 
                                                test_size=.33)

# Fit and transform the data
X_train=data_cleaning(X_train)
X_val=data_cleaning(X_val)
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size=len(tokenizer.word_index)+1
print(f'Vocab Size: {vocab_size}')
X_train=pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
X_val=pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=100)

# Transform the test data
X_test=data_cleaning(X_test)
X_test=pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

X_test2=df_VADER_test2_samp['content_lemma']
y_test2=df_VADER_test2_samp[['sentiment_cat','sentiment']]
#X_test2=X_test2.head(1000)
#y_test2=y_test2.head(1000)
X_test2=data_cleaning(X_test2)
X_test2=pad_sequences(tokenizer.texts_to_sequences(X_test2), maxlen=100)

y_train_reg=y_train['sentiment']
y_val_reg=y_val['sentiment']

#%%

# Hyperparameter tuning on LSTM model
n_units = [128,256,512]
dropouts = [0,.1,.2]
n_hiddenlayers = [1,2,3]
n_epochs = [25,50,75]
learning_rates = [.001,.01,.1]

for units in n_units:
    for dropout in dropouts:
        for hiddenlayers in n_hiddenlayers:
            for epochs in n_epochs:
                for learning_rate in learning_rates:
                    print(units)
                    print(dropout)
                    print(hiddenlayers)
                    print(epochs)
                    print(learning_rate)
                    print('')
                    
                    start = datetime.now()
                    
                    model_reg=Sequential()
                    model_reg.add(layers.Embedding(input_dim=vocab_size,\
                                               output_dim=100,\
                                               input_length=100))
                    if hiddenlayers == 1:
                        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout)))
                    if hiddenlayers == 2:
                        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout, return_sequences=True)))
                        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout)))
                    if hiddenlayers == 3:
                        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout, return_sequences=True)))
                        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout, return_sequences=True)))
                        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout)))
                    model_reg.add(layers.Dense(1))
                    model_reg.compile(optimizer=Adam(learning_rate=learning_rate),\
                                  loss='mse',\
                                  metrics='mae')
                    model_reg.fit(X_train,\
                              y_train_reg,\
                              batch_size=256,\
                              epochs=epochs,\
                              #epochs=2,\
                              validation_data=(X_val,y_val_reg))
                    model_reg_timestamp = str(datetime.now()).replace(' ','_').replace(':','')
                    print(model_reg_timestamp)
                    
                    # Create folder for model
                    if not os.path.exists('{0}/{1}'.format(basefile,model_reg_timestamp)):
                        os.makedirs('{0}/{1}'.format(basefile,model_reg_timestamp))
                    
                    # Save model
                    model_reg.save('{0}/{1}/model_reg.h5'.format(basefile,
                                     model_reg_timestamp))
                    # Save history
                    np.save('{0}/{1}/model_reg_history'.format(basefile,model_reg_timestamp),
                            model_reg.history.history)
                    
                    # Save tokenizer
                    with open('{0}/{1}/tokenizer.pickle'.format(basefile,model_reg_timestamp),
                              'wb') as handle:
                        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    # Plot performance by number of epochs
                    df_sns = pd.DataFrame(model_reg.history.history)
                    df_sns['epoch'] = df_sns.index + 1
                    df_sns_melt = pd.melt(df_sns, id_vars='epoch')
                    
                    df_sns_melt['dataset'] = df_sns_melt.apply(lambda row: get_dataset(row), axis=1)
                    df_sns_melt['stat'] = df_sns_melt['variable'].str.split('_').str[-1]
                    
                    g = sns.relplot(data=df_sns_melt, x='epoch', y='value', hue='dataset', col='stat',
                                kind='line')
                    g.set_axis_labels(x_var = 'Epoch', y_var = 'Value')
                    g.set_titles(col_template = '{col_name}')
                    for ax in g.axes.flat:
                        ax.xaxis.set_major_locator(tkr.MultipleLocator(5))
                    g.tight_layout()
                    g.savefig('{0}/{1}/stat_graphs.png'.format(basefile,model_reg_timestamp))
                    
                    # Test on test dataset from high/very high confidence scores
                    predict_test_reg = model_reg.predict(X_test)
                    y_test_reg = y_test['sentiment']
                    
                    # Test on low/very low confidence tweets                
                    predict_test2_reg = model_reg.predict(X_test2)
                    y_test2_reg = y_test2['sentiment']
                    
                    end = datetime.now()
                    print(str(end-start))
                    
                    score_dict_reg = {'model': model_reg_timestamp,
                                      'units': units,
                                      'dropout': dropout,
                                      'hiddenlayers': hiddenlayers,
                                      'epochs': epochs,
                                      'learning_rate': learning_rate,
                                      'time_taken': str(end-start),
                                      'test_mse':mean_squared_error(y_test_reg,predict_test_reg),
                                      'test_mae':median_absolute_error(y_test_reg,
                                                                       predict_test_reg),
                                      'test_corr':y_test.reset_index()['sentiment'].corr(
                                              pd.Series(predict_test_reg[:,0])),                                                   
                                      'test2_mse':mean_squared_error(y_test2_reg,
                                                                     predict_test2_reg),
                                      'test2_mae':median_absolute_error(y_test2_reg,
                                                                        predict_test2_reg),
                                      'test2_corr':y_test2.reset_index()['sentiment'].corr(
                                              pd.Series(predict_test2_reg[:,0]))
                                      }
                    
                    df_scores = df_scores.append(score_dict_reg, ignore_index=True)
                    
                    np.save('{0}/{1}/score_dict'.format(basefile,model_reg_timestamp),
                            score_dict_reg)

df_scores.to_csv('{0}/df_scores_{1}.csv'.format(basefile,str(datetime.now()).replace(' ','_').replace(':','')))

## Get results from all hyperparameter combinations into dataframe
# Define common string in folder name
folderstring1 = '2022-02-09_'
folderstring2 = '2022-02-10_'
folderstring3 = '2022-02-11_'
folderstring4 = '2022-02-12_'
folderstring5 = '2022-02-13_'
folderstring6 = '2022-02-14_'
folderstring7 = '2022-02-16_'
folderstring8 = '2022-02-17_'
folderstring9 = '2022-02-18_'
# Get list of model timestamps
all_folders = listdir(filepath + "\INDP\Models\model_reg")
folders = [s for s in all_folders if (folderstring1 in s) or 
           (folderstring2 in s) or (folderstring3 in s) or (folderstring4 in s)
            or (folderstring5 in s) or (folderstring6 in s) or 
            (folderstring7 in s) or (folderstring8 in s) or 
            (folderstring9 in s)]

# Loop through folders and append results to dataframe
df_scores = pd.DataFrame()
for f in folders:
    print(f)
    score_dict = np.load('{0}/{1}/score_dict.npy'.format(basefile,f),allow_pickle='TRUE').item()
    score_dict['model'] = f
    df_scores = df_scores.append(score_dict, ignore_index=True)

print(pd.value_counts(df_scores.dropout))
print(pd.value_counts(df_scores.epochs))
print(pd.value_counts(df_scores.hiddenlayers))
print(pd.value_counts(df_scores.learning_rate))
print(pd.value_counts(df_scores.units))

#%% Run a few candidate models on bigger samples

df_VADER_train_samp = df_VADER_train.sample(n=10000)
df_VADER_test2_samp = df_VADER_test2.sample(n=10000)
X=df_VADER_train_samp['content_lemma']
y=df_VADER_train_samp[['sentiment_cat','sentiment']]
#X=X.head(1000)
#y=y.head(1000)
X_train, X_testval, y_train, y_testval=train_test_split(X, y, test_size=.3)
X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, 
                                                test_size=.33)

# Fit and transform the data
X_train=data_cleaning(X_train)
X_val=data_cleaning(X_val)
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size=len(tokenizer.word_index)+1
print(f'Vocab Size: {vocab_size}')
X_train=pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
X_val=pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=100)

# Transform the test data
X_test=data_cleaning(X_test)
X_test=pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

X_test2=df_VADER_test2_samp['content_lemma']
y_test2=df_VADER_test2_samp[['sentiment_cat','sentiment']]
#X_test2=X_test2.head(1000)
#y_test2=y_test2.head(1000)
X_test2=data_cleaning(X_test2)
X_test2=pad_sequences(tokenizer.texts_to_sequences(X_test2), maxlen=100)

y_train_reg=y_train['sentiment']
y_val_reg=y_val['sentiment']

# Chose top 5 based on MSE
df_hp = pd.DataFrame([[0.1,75,3,0.001,128],
                      [0.2,50,3,0.01,128],
                      [0.2,75,2,0.001,128],
                      [0.2,75,1,0.001,256],
                      [0.0,75,2,0.001,256]],
                        columns=['dropout','epochs','hiddenlayers',
                                 'learning_rate','units'])

for index,row in df_hp.iterrows():
    print(row['dropout'])
    print(row['epochs'])
    print(row['hiddenlayers'])
    print(row['learning_rate'])
    print(row['units'])
    print('')
    
    dropout = row['dropout']
    epochs = row['epochs'].astype('int')
    hiddenlayers = row['hiddenlayers']
    learning_rate = row['learning_rate']
    units = row['units'].astype('int')
    
    start = datetime.now()
    
    model_reg=Sequential()
    model_reg.add(layers.Embedding(input_dim=vocab_size,\
                               output_dim=100,\
                               input_length=100))
    if hiddenlayers == 1:
        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout)))
    if hiddenlayers == 2:
        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout, return_sequences=True)))
        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout)))
    if hiddenlayers == 3:
        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout, return_sequences=True)))
        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout, return_sequences=True)))
        model_reg.add(layers.Bidirectional(layers.LSTM(units=units, dropout=dropout)))
    model_reg.add(layers.Dense(1))
    model_reg.compile(optimizer=Adam(learning_rate=learning_rate),\
                  loss='mse',\
                  metrics='mae')
    model_reg.fit(X_train,\
              y_train_reg,\
              batch_size=256,\
              epochs=epochs,\
              #epochs=2,\
              validation_data=(X_val,y_val_reg))
    model_reg_timestamp = str(datetime.now()).replace(' ','_').replace(':','')
    print(model_reg_timestamp)
    
    # Create folder for model
    if not os.path.exists('{0}/{1}'.format(basefile,model_reg_timestamp)):
        os.makedirs('{0}/{1}'.format(basefile,model_reg_timestamp))
    
    # Save model
    model_reg.save('{0}/{1}/model_reg.h5'.format(basefile,
                     model_reg_timestamp))
    # Save history
    np.save('{0}/{1}/model_reg_history'.format(basefile,model_reg_timestamp),
            model_reg.history.history)
    
    # Save tokenizer
    with open('{0}/{1}/tokenizer.pickle'.format(basefile,model_reg_timestamp),
              'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Plot performance by number of epochs
    df_sns = pd.DataFrame(model_reg.history.history)
    df_sns['epoch'] = df_sns.index + 1
    df_sns_melt = pd.melt(df_sns, id_vars='epoch')
    
    df_sns_melt['dataset'] = df_sns_melt.apply(lambda row: get_dataset(row), axis=1)
    df_sns_melt['stat'] = df_sns_melt['variable'].str.split('_').str[-1]
    
    g = sns.relplot(data=df_sns_melt, x='epoch', y='value', hue='dataset', col='stat',
                kind='line')
    g.set_axis_labels(x_var = 'Epoch', y_var = 'Value')
    g.set_titles(col_template = '{col_name}')
    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(tkr.MultipleLocator(5))
    g.tight_layout()
    g.savefig('{0}/{1}/stat_graphs.png'.format(basefile,model_reg_timestamp))
    
    # Test on test dataset from high/very high confidence scores
    predict_test_reg = model_reg.predict(X_test)
    y_test_reg = y_test['sentiment']
    
    # Test on low/very low confidence tweets                
    predict_test2_reg = model_reg.predict(X_test2)
    y_test2_reg = y_test2['sentiment']
    
    end = datetime.now()
    print(str(end-start))
    
    score_dict_reg = {'model': model_reg_timestamp,
                      'units': units,
                      'dropout': dropout,
                      'hiddenlayers': hiddenlayers,
                      'epochs': epochs,
                      'learning_rate': learning_rate,
                      'time_taken': str(end-start),
                      'test_mse':mean_squared_error(y_test_reg,predict_test_reg),
                      'test_mae':median_absolute_error(y_test_reg,
                                                       predict_test_reg),
                      'test_corr':y_test.reset_index()['sentiment'].corr(
                              pd.Series(predict_test_reg[:,0])),                                                   
                      'test2_mse':mean_squared_error(y_test2_reg,
                                                     predict_test2_reg),
                      'test2_mae':median_absolute_error(y_test2_reg,
                                                        predict_test2_reg),
                      'test2_corr':y_test2.reset_index()['sentiment'].corr(
                              pd.Series(predict_test2_reg[:,0]))
                      }
    
    df_scores = df_scores.append(score_dict_reg, ignore_index=True)
    
    np.save('{0}/{1}/score_dict'.format(basefile,model_reg_timestamp),
            score_dict_reg)




















