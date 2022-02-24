# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 09:58:35 2022

@author: Joe.WozniczkaWells
"""

import pandas as pd
import numpy as np
import os

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords as sw
from nltk.tag import pos_tag
import pickle
from os import chdir, getcwd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#%% Read in data
# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

basefile = 'INDP/Models'
model_multi_timestamp = '2022-02-05_205022.742078'
model_reg_timestamp = '2022-02-18_175759.443774'

df_VADER = pd.read_csv("INDP//Data//df_VADER.csv")

#%% Pre-process data
# Format tweet_id
df_VADER['tweet_id'] = df_VADER['tweet_id'].astype('Int64').apply(str)

# Categorise sentiment (must be 0, 1 and 2 for to_categorical to work)
def cat_sentiment(row,var):
    # Positive
    if row[var] >= 0.05:
        return 1
    # Negative
    elif row[var] <= -0.05:
        return -1
    # Neutral
    else:
        return 0

def cat_sentiment_str(row,var):
    # Positive
    if row[var] >= 0.05:
        return 'Positive'
    # Negative
    elif row[var] <= -0.05:
        return 'Negative'
    # Neutral
    else:
        return 'Neutral'

def cat_setconf(row):
    if row['sentconf'] >= thr:
        return 'Very High'
    elif row['sentconf'] >= thr - 0.5 * std:
        return 'High'
    elif row['sentconf'] >= thr - std:
        return 'Low'
    elif row['sentconf'] > 0:
        return 'Very Low'
    else:
        return 'Zero'

# Code taken from https://towardsdatascience.com/sentiment-analysis-comparing-3-common-approaches-naive-bayes-lstm-and-vader-ab561f834f89

def data_cleaning(text_list):
    stopwords_rem=False
    stopwords_en=sw.words('english')
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

# Read in tokenizers used in model training
with open('{0}/model_multi/{1}/tokenizer.pickle'.format(basefile,
          model_multi_timestamp), 'rb') as handle:
    tokenizer_multi = pickle.load(handle)

with open('{0}/model_reg/{1}/tokenizer.pickle'.format(basefile,
          model_reg_timestamp), 'rb') as handle:
    tokenizer_reg = pickle.load(handle)

df_VADER['sentiment_cat'] = df_VADER.apply(lambda row: cat_sentiment(row,'sentiment'), axis=1)

# Categorise sentiment confidence
mcs = df_VADER.sentconf.mean()
std = df_VADER.sentconf.std()
thr = mcs + 0.5 * std

df_VADER['sentconf_cat'] = df_VADER.apply(lambda row: cat_setconf(row), axis=1)

X=df_VADER[df_VADER.content_lemma.notna()].reset_index()['content_lemma']
X_multi=data_cleaning(X.head(1000))
X_multi=pad_sequences(tokenizer_multi.texts_to_sequences(X_multi), maxlen=100)
X_reg=data_cleaning(X)
X_reg=pad_sequences(tokenizer_reg.texts_to_sequences(X_reg), maxlen=100)

#%% Score out on all England data
# Load models
model_multi = load_model('{0}/model_multi/{1}/model_multi.h5'.format(basefile,
                         model_multi_timestamp))
model_reg = load_model('{0}/model_reg/{1}/model_reg.h5'.format(basefile,
                       model_reg_timestamp))

# Score out
pred_multi = model_multi.predict_classes(X_multi)
#classes_multi = np.argmax(pred_multi,axis=1)
pred_reg = model_reg.predict(X_reg)

# Add to dataframe, to help with plotting graphs
df_output = df_VADER[df_VADER.content_lemma.notna()].reset_index().copy()
df_output['LSTM_multi'] = pred_multi-1
df_output['LSTM_reg'] = pred_reg
df_output['LSTM_reg_cat'] = df_output.apply(lambda row: cat_sentiment(row,'LSTM_reg'), axis=1)

#%% Start plotting

sentconf_cat_order = ['Very High','High','Low','Very Low','Zero']

# Compare the outputs of the two models

## All England tweets
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.scatterplot(data=df_output, x='LSTM_reg', y='LSTM_multi', ax=ax)
ax.set(xlabel = 'LSTM regression score ({0})'.format(model_reg_timestamp), 
       ylabel = 'LSTM class ({0})'.format(model_multi_timestamp),
       title = 'Comparison of regression and classification LSTM models')
plt.tight_layout()
plt.savefig("INDP/Models/LSTM_reg_v_multi_{0}_{1}.png".format(
        model_reg_timestamp,model_multi_timestamp))

## Faceted by VADER confidence score
g = sns.relplot(data=df_output, x='LSTM_reg', y='LSTM_multi', 
                col='sentconf_cat', col_order=sentconf_cat_order)
g.set_axis_labels(x_var = 'LSTM regression score ({0})'.format(model_reg_timestamp), 
                  y_var = 'LSTM class ({0})'.format(model_multi_timestamp))
g.set_titles(col_template = 'Confidence in VADER score: {col_name}')
g.fig.suptitle('Comparison of regression and classification LSTM models,'
               ' by VADER confidence category')
g.tight_layout()
g.savefig("INDP/Models/LSTM_reg_v_multi_{0}_{1}_sentconf_cat.png".format(
        model_reg_timestamp,model_multi_timestamp))

# Compare to VADER scores

## Compare distributions
fig, ax = plt.subplots(1,2, figsize = (12,6))
fig = sns.violinplot(data=df_output[['LSTM_reg','sentiment']],cut=0,
                     inner='quartile',ax=ax[0])
ax[0].set_xticklabels(['LSTM','VADER'])
fig = sns.violinplot(data=df_output[['LSTM_multi','sentiment_cat']],cut=0,
                     inner='quartile',ax=ax[1])
ax[1].set_xticklabels(['LSTM','VADER'])
plt.suptitle('Comparison of LSTM and VADER distributions')
plt.tight_layout()
plt.savefig("INDP/Models/dists_LSTM_VADER_{0}_{1}.png".format(
        model_reg_timestamp,model_multi_timestamp))

## Compare LSTM scores/classes to VADER ones
fig, ax = plt.subplots(1,2, figsize = (12,6))
fig = sns.scatterplot(data=df_output, x='LSTM_reg', y='sentiment', 
                      hue='sentconf_cat', 
                      hue_order=sentconf_cat_order, 
                      ax=ax[0], legend=False)
ax[0].set(xlabel = 'LSTM regression score ({0})'.format(model_reg_timestamp),
  ylabel = 'VADER sentiment score')
fig = sns.scatterplot(data=df_output, x='LSTM_multi', y='sentiment',
                      hue='sentconf_cat', 
                      hue_order=sentconf_cat_order, 
                      ax=ax[1])
ax[1].set(xlabel = 'LSTM class ({0})'.format(model_multi_timestamp),
  ylabel = 'VADER sentiment score')
plt.legend(bbox_to_anchor=(1.05,1),loc='upper left',borderaxespad=0,
           title='VADER sentiment \nconfidence score')
plt.suptitle('Comparison of LSTM predictions and VADER scores')
plt.tight_layout()
plt.savefig("INDP/Models/LSTM_v_VADER_{0}_{1}.png".format(
        model_reg_timestamp,model_multi_timestamp))

### What about by sentconf_cat
#### Regression
g = sns.relplot(data=df_output, x='LSTM_reg', y='sentiment', 
                col='sentconf_cat', col_order=sentconf_cat_order)
g.set_axis_labels(x_var = 'LSTM regression score ({0})'.format(model_reg_timestamp), 
                  y_var = 'VADER sentiment score')
g.set_titles(col_template = 'Confidence in VADER score: {col_name}')
g.fig.suptitle('Comparison of regression LSTM score and VADER score, by '
               'VADER confidence category')
g.tight_layout()
g.savefig("INDP/Models/LSTM_reg_v_VADER_{0}_sentconf_cat.png".format(
        model_reg_timestamp,model_multi_timestamp))

#### Multiclass
g = sns.relplot(data=df_output, x='LSTM_multi', y='sentiment', 
                col='sentconf_cat', col_order=sentconf_cat_order)
g.set_axis_labels(x_var = 'LSTM class ({0})'.format(model_multi_timestamp), 
                  y_var = 'VADER score')
g.set_titles(col_template = 'Confidence in VADER score: {col_name}')
g.fig.suptitle('Comparison of multiclass LSTM class and VADER score, by '
               'VADER confidence category')
g.tight_layout()
g.savefig("INDP/Models/LSTM_multi_v_VADER_{0}_sentconf_cat.png".format(
        model_multi_timestamp))

## Plot absolute error against sentconf
df_output['absolute_error'] = abs(df_output['LSTM_reg']-df_output['sentiment'])

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.scatterplot(data=df_output, x='absolute_error', y='sentconf')
ax.set(xlabel = 'Absolute difference between regression LSTM score and VADER'
           'score ({0})'.format(model_reg_timestamp), 
       ylabel = 'Confidence in VADER score',
       title = 'Comparison of absolute error and VADER confidence')
plt.savefig("INDP/Models/ae_v_sentconf_{0}.png".format(
        model_reg_timestamp))

## Word clouds by sentiment_cat
# Wordcloud with stop words excluded
stopwords_wc=set(STOPWORDS)
# Exclude a few other meaningless words
stopwords_wc.update(['https','co','t','s','amp','u'])

fig_reg, ax_reg = plt.subplots(1,3, figsize = (24,12))
fig_multi, ax_multi = plt.subplots(1,3, figsize = (24,12))
for i,cat in enumerate(['Positive','Neutral','Negative']):
    #Regression
    df = df_output[df_output.apply(lambda row: cat_sentiment_str(row,'LSTM_reg'), axis=1)==cat]
    #df = df_output[df_output['LSTM_reg_cat']==cat]
    word_list = ' '.join(df['tweet_text'].tolist())
    
    wordcloud_excstopwords = WordCloud(stopwords=stopwords_wc,
                                   background_color="white",
                                   max_words=len(word_list),max_font_size=40,
                                   relative_scaling=.5).generate(word_list)
    #plt.figure()
    ax_reg[i].imshow(wordcloud_excstopwords)
    ax_reg[i].axis("off")
    ax_reg[i].set(title='LSTM regression: {0}'.format(cat))
    
    #Multiclass
    df = df_output[df_output.apply(lambda row: cat_sentiment_str(row,'LSTM_multi'), axis=1)==cat]
    word_list = ' '.join(df['tweet_text'].tolist())
    
    wordcloud_excstopwords = WordCloud(stopwords=stopwords_wc,
                                   background_color="white",
                                   max_words=len(word_list),max_font_size=40,
                                   relative_scaling=.5).generate(word_list)
    #plt.figure()
    ax_multi[i].imshow(wordcloud_excstopwords)
    ax_multi[i].axis("off")
    ax_multi[i].set(title='LSTM classification: {0}'.format(cat))
    
fig_multi.savefig('INDP/Models/LSTM_multi_wordcloud_excstopwords_{0}_{1}.png'.format(cat,model_multi_timestamp))
fig_reg.savefig('INDP/Models/LSTM_reg_wordcloud_excstopwords_{0}_{1}.png'.format(cat,model_reg_timestamp))


#%%
## Shapley plots - may have to revert to tensorflow 2.5.0 to fix this. Did this on 5/2/22 so try again tomorrow.
##
#X_reg=data_cleaning(X.head(50))
#X_reg=pad_sequences(tokenizer_reg.texts_to_sequences(X_reg), maxlen=100)
##
#words = tokenizer_reg.sequences_to_texts(X_reg)
words = tokenizer_reg.word_index
word_lookup = list()
for i in words.keys():
    word_lookup.append(i)

word_lookup = [''] + word_lookup

num_explanations = 10
explainer = shap.DeepExplainer(model_reg,X_reg[0:100])
#explainer = shap.DeepExplainer(model_reg,X_reg)
shap_values = explainer.shap_values(X_reg[200:200+num_explanations])
#shap_values = explainer.shap_values(X_reg) #here 20/2/22
shap.summary_plot(shap_values[0], word_lookup, show=False)
shap.force_plot(explainer.expected_value, shap_values[0], X_reg[1000])
plt.savefig('shap_summary_plot2.png')

import sage
tf.experimental.numpy.experimental_enable_numpy_behavior(
    prefer_float32=False
)

imputer = sage.MarginalImputer(model_reg, X_reg[:1000])
estimator = sage.PermutationEstimator(imputer,'mse')

sage_values = estimator(X_reg,pred_reg)

# Multi
words = tokenizer_multi.word_index
word_lookup = list()
for i in words.keys():
    word_lookup.append(i)

word_lookup = [''] + word_lookup

num_explanations = 10
explainer = shap.DeepExplainer(model_multi,X_multi[:100])
shap_values = explainer.shap_values(X_multi[:num_explanations])
shap.summary_plot(shap_values[0], word_lookup, show=False)
plt.savefig('shap_summary_plot2.png')

# Try on more records, from shap.readthedocs
def f(X):
    return model_reg.predict([X_reg[:,i] for i in range(X_reg.shape[1])]).flatten()

explainer = shap.KernelExplainer(f, X_reg.iloc[:50,:])


shap.force_plot(explainer.expected_value[0], shap_values[0][0], feature_names=X_reg.columns)


