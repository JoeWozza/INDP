# -*- coding: utf-8 -*-
"""
This code was used to create initial versions of the LSTM models. A multiclass
model and a regression model were trained and the outputs investigated.

@author: Joe Wozniczka-Wells
"""

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tkr
from datetime import datetime
import pickle
import numpy as np
import os
from os import chdir, getcwd

# Set file path
filepath = ("C:\\Users\\joew\\Documents\\Apprenticeship\\UoB\\SPFINDP21T4\\")
chdir(filepath)

# Create model folder
if not os.path.exists('INDP/Models'):
    os.makedirs('INDP/Models')
if not os.path.exists('INDP/Models/Initial_models'):
    os.makedirs('INDP/Models/Initial_models')
if not os.path.exists('INDP/Models/Initial_models/model_reg'):
    os.makedirs('INDP/Models/Initial_models/model_reg')
if not os.path.exists('INDP/Models/Initial_models/model_multi'):
    os.makedirs('INDP/Models/Initial_models/model_multi')

df_VADER = pd.read_csv("INDP/Data/VADER/df_VADER.csv")

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

# Code adapted from https://towardsdatascience.com/sentiment-analysis-comparing-3-common-approaches-naive-bayes-lstm-and-vader-ab561f834f89

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
n_epochs = 100

# Remove neutrals
#df_VADER = df_VADER[df_VADER.sentiment_cat >= 0].reset_index()
#df_VADER_train.to_csv("df_VADER_train.csv")

# Break data down into a training set and a validation set. AUC cannot be
# calculated if y_true doesn't contain all casses. There are so few neutrals
# that I don't think it's worth including these in the multiclass anyway.
# Actually, that isn't the case. Neutrals make up about 20% of the dataset, so
# I should keep them in.
#X=df_VADER_train.reset_index()['content_lemma']
#y=df_VADER_train.reset_index()[['sentiment_cat','sentiment']]
X=df_VADER_train['content_lemma']
y=df_VADER_train[['sentiment_cat','sentiment']]
#X=X.head(10000)
#y=y.head(10000)
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
y_train_cat=to_categorical(y_train['sentiment_cat'])
y_val_cat=to_categorical(y_val['sentiment_cat'])

# Create an LSTM model with an Embedding layer and fit training data
model_multi=Sequential()
model_multi.add(layers.Embedding(input_dim=vocab_size,\
                           output_dim=100,\
                           input_length=100))
model_multi.add(layers.Bidirectional(layers.LSTM(128)))
model_multi.add(layers.Dense(3,activation='softmax'))
model_multi.compile(optimizer='adam',\
              loss='categorical_crossentropy',\
              metrics=['accuracy','AUC'])
model_multi.fit(X_train,\
          y_train_cat,\
          batch_size=256,\
          epochs=n_epochs,\
          #epochs=2,\
          validation_data=(X_val,y_val_cat))
model_multi_timestamp = str(datetime.now()).replace(' ','_').replace(':','')

# Create folder for model
basefile = 'INDP/Models/model_multi'
if not os.path.exists('{0}/{1}'.format(basefile,model_multi_timestamp)):
    os.makedirs('{0}/{1}'.format(basefile,model_multi_timestamp))

# Save model
model_multi.save('{0}/{1}/model_multi.h5'.format(basefile,
                 model_multi_timestamp))
# Save history
np.save('{0}/{1}/model_multi_history'.format(basefile,model_multi_timestamp),
        model_multi.history.history)

# Save tokenizer
with open('{0}/{1}/tokenizer.pickle'.format(basefile,model_multi_timestamp),
          'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

## Load model
#model_multi_ = load_model('{0}/{1}/model_multi.h5'.format(basefile,
#                          model_multi_timestamp))
## Load history
#history=np.load('{0}/{1}/model_multi_history.npy'.format(basefile,
#                model_multi_timestamp),allow_pickle='TRUE').item()

# Plot performance by number of epochs
df_sns = pd.DataFrame(model_multi.history.history)
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
g.savefig('{0}/{1}/stat_graphs.png'.format(basefile,model_multi_timestamp))

# Calculate accuray and AUC on test data
# First need to work out how to process test data. Can it be done identically
# to train and validation data? Are the word-to-number lookups specific to the
# case? Specific to case but saved in tokenizer, so can be applied to test data.

#X_test=df_VADER_test['content_lemma']
#y_test=df_VADER_test['sentiment_cat']

# Test on test dataset from high/very high confidence scores
# Transform the data
X_test=data_cleaning(X_test)
X_test=pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)
y_test_cat=to_categorical(y_test['sentiment_cat'])

# Score
predict_test = model_multi.predict(X_test)
predict_test_cat = np.around(predict_test)
print(predict_test_cat.sum(axis=0))

print(roc_auc_score(y_test_cat,predict_test)) #0.8567052301184198
print(accuracy_score(y_test_cat,predict_test_cat)) #0.9240584482403786

# Test on low/very low confidence tweets
X_test2=df_VADER_test2['content_lemma']
y_test2=df_VADER_test2[['sentiment_cat','sentiment']]
#X_test2=X_test2.head(100)
#y_test2=y_test2.head(100)
X_test2=data_cleaning(X_test2)
X_test2=pad_sequences(tokenizer.texts_to_sequences(X_test2), maxlen=100)
y_test2_cat=to_categorical(y_test2['sentiment_cat'])

# Score
predict_test2 = model_multi.predict(X_test2)
predict_test2_cat = np.around(predict_test2)
print(predict_test2_cat.sum(axis=0))

print(roc_auc_score(y_test2_cat,predict_test2)) #0.6430047667939832
print(accuracy_score(y_test2_cat,predict_test2_cat)) #0.6008560997104369

score_dict_multi = {'test_auc':roc_auc_score(y_test_cat,predict_test),
                    'test_acc':accuracy_score(y_test_cat,predict_test_cat),
                    'test2_auc':roc_auc_score(y_test2_cat,predict_test2),
                    'test2_acc':accuracy_score(y_test2_cat,predict_test2_cat)}

np.save('{0}/{1}/score_dict'.format(basefile,model_multi_timestamp),
        score_dict_multi)

# Seems to work quite well. Wonder if I can predict scores rather than categories, or at least multiple categories.

#%% Regression

#df_VADER = pd.read_csv("INDP//Data//df_VADER.csv")

# Format tweet_id
#df_VADER['tweet_id'] = df_VADER['tweet_id'].astype('Int64').apply(str)

# Categorise sentiment confidence
#mcs = df_VADER.sentconf.mean()
#std = df_VADER.sentconf.std()
#thr = mcs + 0.5 * std

#df_VADER['sentconf_cat'] = df_VADER.apply(lambda row: cat_setconf(row), axis=1)

#df_VADER_train = df_VADER[df_VADER['sentconf'] >= thr - 0.5 * std].reset_index()
#df_VADER_test2 = df_VADER[(df_VADER['sentconf'] < thr - 0.5 * std) & 
#                         (df_VADER['sentconf'] > 0)].reset_index()

#X=df_VADER_train['content_lemma']
#y=df_VADER_train['sentiment']
#X=X.head(100)
#y=y.head(100)
#X_train, X_testval, y_train, y_testval=train_test_split(X, y, test_size=.3)
#X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, 
#                                                test_size=.33)

# Fit and transform the data
#X_train=data_cleaning(X_train)
#X_val=data_cleaning(X_val)
#tokenizer=Tokenizer()
#tokenizer.fit_on_texts(X_train)
#vocab_size=len(tokenizer.word_index)+1
#print(f'Vocab Size: {vocab_size}')
#X_train=pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
#X_val=pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=100)

y_train_reg=y_train['sentiment']
y_val_reg=y_val['sentiment']

# Create an LSTM model with an Embedding layer and fit training data
model_reg=Sequential()
model_reg.add(layers.Embedding(input_dim=vocab_size,\
                           output_dim=100,\
                           input_length=100))
model_reg.add(layers.Bidirectional(layers.LSTM(128)))
model_reg.add(layers.Dense(1))
model_reg.compile(optimizer='adam',\
              loss='mse',\
              metrics='mae')
model_reg.fit(X_train,\
          y_train_reg,\
          batch_size=256,\
          epochs=n_epochs,\
          #epochs=2,\
          validation_data=(X_val,y_val_reg))
model_reg_timestamp = str(datetime.now()).replace(' ','_').replace(':','')

# Create folder for model
basefile = 'INDP/Models/model_reg'
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

# Score on test data (already cleaned in multiclass model)
predict_test_reg = model_reg.predict(X_test)
y_test_reg = y_test['sentiment']

print(y_test.reset_index()['sentiment'].corr(pd.Series(predict_test_reg[:,0]))) #0.8915527796798747
print(mean_squared_error(y_test_reg,predict_test_reg)) #0.07363959430258582
print(median_absolute_error(y_test_reg,predict_test_reg)) #0.14863662786483767

predict_test2_reg = model_reg.predict(X_test2)
y_test2_reg = y_test2['sentiment']

print(y_test2.reset_index()['sentiment'].corr(pd.Series(predict_test2[:,0]))) #-0.3999251015584579 (!)
print(mean_squared_error(y_test2_reg,predict_test2_reg)) #0.17094737939809393
print(median_absolute_error(y_test2_reg,predict_test2_reg)) #0.2536297626972198

score_dict_reg = {'test_mse':mean_squared_error(y_test_reg,predict_test_reg),
                  'test_mae':median_absolute_error(y_test_reg,
                                                   predict_test_reg),
                  'test_corr':y_test.reset_index()['sentiment'].corr(
                          pd.Series(predict_test_reg[:,0])),                                                   
                  'test2_mse':mean_squared_error(y_test2_reg,
                                                 predict_test2_reg),
                  'test2_mae':median_absolute_error(y_test2_reg,
                                                    predict_test2_reg),
                  'test2_corr':y_test2.reset_index()['sentiment'].corr(
                          pd.Series(predict_test2[:,0]))
                  }

np.save('{0}/{1}/score_dict'.format(basefile,model_reg_timestamp),
        score_dict_reg)