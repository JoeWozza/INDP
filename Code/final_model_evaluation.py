# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:23:28 2022

@author: Joe.WozniczkaWells
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.ticker as tkr
from datetime import datetime
import numpy as np
import os
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from os import chdir, listdir

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

from INDP.Code import LSTM
class_lstm = LSTM.LSTM()
from INDP.Code import VADER
class_v = VADER.VADER()

df_LSTM_sent = pd.read_csv("INDP\Data\df_LSTM_sent.csv")
# Categorise sentiment score
df_LSTM_sent['LSTM_sent_cat'] = df_LSTM_sent.apply(lambda row: 
    class_lstm.cat_sentiment_str(row['LSTM_sent']), axis=1)
df_LSTM_sent_unique = df_LSTM_sent.drop_duplicates(subset=['tweet_id'])

#%% Compare to VADER

# Load packages
import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
analyzer = SentimentIntensityAnalyzer()

# Define stopwords list
stopwords_fin = class_v.fin_stopwords(stopwords.words('english'))
# Sort out tweet_id field
df_LSTM_sent_unique['tweet_id'] = (df_LSTM_sent_unique['tweet_id']
    .astype('Int64').apply(str))
# Clean, tokenize and lemmatise tweet content
df_LSTM_sent_unique['content_lemma'] = (df_LSTM_sent_unique['tweet_text']
    .apply(class_v.clean)
    .apply(class_v.tokenize)
    .apply(class_v.lemmatise, stopwords_list=stopwords_fin))
# Calculate sentiment
df_LSTM_sent_unique['VADER_sent'] = (df_LSTM_sent_unique['content_lemma']
    .apply(class_v.VADERsentiment))
# Calculate sentiment confidence
df_LSTM_sent_unique['VADER_conf'] = (df_LSTM_sent_unique['content_lemma']
    .apply(class_v.sentconf))
# Categorise sentiment score
df_LSTM_sent_unique['VADER_sent_cat'] = df_LSTM_sent_unique.apply(lambda row: 
    class_v.cat_sentiment(row['VADER_sent']), axis=1)
# Categorise sentiment confidence score
mean_cs,std_cs = class_v.cat_sentconf_stats(df_LSTM_sent_unique['VADER_conf'])
df_LSTM_sent_unique['VADER_conf_cat'] = df_LSTM_sent_unique.apply(lambda row:
    class_v.cat_sentconf(row['VADER_conf'],mean_cs,std_cs), axis=1)

sentconf_cat_order = ['VeryHigh','High','Low','VeryLow','Zero']

# Compare to VADER scores

## Compare distributions
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.violinplot(data=df_LSTM_sent_unique[['LSTM_sent','VADER_sent']],
                     cut=0,inner='quartile',ax=ax)
ax.set_xticklabels(['LSTM','VADER'])
plt.suptitle('Comparison of LSTM and VADER distributions')
plt.tight_layout()
plt.savefig("INDP/Images/dists_LSTM_VADER.png")

## Compare LSTM scores/classes to VADER ones
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.scatterplot(data=df_LSTM_sent_unique,x='LSTM_sent',y='VADER_sent', 
                      hue='VADER_conf_cat',hue_order=sentconf_cat_order, 
                      ax=ax)
ax.set(xlabel = 'LSTM sentiment score',ylabel = 'VADER sentiment score')
plt.suptitle('Comparison of LSTM predictions and VADER scores')
plt.tight_layout()
plt.savefig("INDP/Images/LSTM_v_VADER.png")

### What about by sentconf_cat
g = sns.relplot(data=df_LSTM_sent_unique, x='LSTM_sent', y='VADER_sent', 
                col='VADER_conf_cat', col_order=sentconf_cat_order)
g.set_axis_labels(x_var = 'LSTM sentiment score', 
                  y_var = 'VADER sentiment score')
g.set_titles(col_template = 'Confidence in VADER score: {col_name}')
g.fig.suptitle('Comparison of LSTM and VADER sentiment scores, by VADER '
               'confidence category')
g.tight_layout()
g.savefig("INDP/Images/LSTM_reg_v_VADER_sentconf_cat.png")

df_LSTM_sent_unique.groupby('VADER_conf_cat')[['LSTM_sent','VADER_sent']].corr()

## Plot absolute error against sentconf
df_LSTM_sent_unique['absolute_error'] = abs(df_LSTM_sent_unique['LSTM_sent']-
                   df_LSTM_sent_unique['VADER_sent'])

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.scatterplot(data=df_LSTM_sent_unique, x='absolute_error', 
                      y='VADER_conf')
ax.set(xlabel = 'Absolute difference between LSTM and VADER sentiment scores', 
       ylabel = 'Confidence in VADER score',
       title = 'Comparison of absolute error and VADER confidence')
plt.tight_layout()
plt.savefig("INDP/Images/ae_v_sentconf.png")

#%% Investigate sentiment scores, overall, by search term and by UTLA

# Plot distribution of sentiment scores
fig, ax = plt.subplots(figsize = (12,6))
#fig = sns.kdeplot(data=df_VADER, x='sentiment', cut=0, ax=ax, color='#007C91')
fig = sns.distplot(df_LSTM_sent_unique['LSTM_sent'], ax=ax, color='#007C91', 
                   kde_kws={'clip': (-1,1)})                
ax.set(xlabel = 'LSTM sentiment score')
plt.tight_layout()
plt.savefig("INDP//Images//LSTM_sentiment.png")

# Plot distribution of sentiment scores by search term
search_terms = (df_LSTM_sent_unique.drop_duplicates(subset=['search_term'])
    ['search_term'].sort_values())
g = sns.FacetGrid(df_LSTM_sent_unique, col = 'search_term', col_wrap = 5, 
                  col_order = search_terms)
g.map(sns.histplot, 'LSTM_sent', color='#007C91', kde=True, 
      kde_kws={'clip': (-1,1)}, linewidth=0)
g.set_axis_labels(x_var = 'LSTM sentiment score', y_var = 'Density')
g.set_titles(col_template = '{col_name}')
g.tight_layout()
g.savefig("INDP//Images//LSTM_sentiment_search_term_hist.png")

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.boxplot(data=df_LSTM_sent_unique, x='search_term', y='LSTM_sent', 
                  color='#007C91', order = search_terms)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.savefig("INDP//Images//LSTM_sentiment_search_term_box.png")

# Plot distribution of sentiment scores by UTLA
utlas = (df_LSTM_sent.drop_duplicates(subset=['area'])
    ['area'].sort_values())
g = sns.FacetGrid(df_LSTM_sent, col = 'area', col_wrap = 5,
                  col_order = utlas)
g.map(sns.histplot, 'LSTM_sent', color='#007C91', kde=True, 
      kde_kws={'clip': (-1,1)}, linewidth=0)
g.set_axis_labels(x_var = 'LSTM sentiment score', y_var = 'Density')
g.set_titles(col_template = '{col_name}')
g.tight_layout()
g.savefig("INDP//Images//LSTM_sentiment_UTLA_hist.png")

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.boxplot(data=df_LSTM_sent, x='area', y='LSTM_sent', 
                  color='#007C91', order = utlas)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.savefig("INDP//Images//LSTM_sentiment_UTLA_box.png")

#%% Word clouds
# Wordcloud with stop words excluded
stopwords_wc=set(STOPWORDS)
# Exclude a few other meaningless words
stopwords_wc.update(['https','co','t','s','amp','u','covid'])
stopwords_wc.update(search_terms)

# By sentiment category
fig, ax = plt.subplots(1,3, figsize = (24,12))
for i,cat in enumerate(['Positive','Neutral','Negative']):
    df = df_LSTM_sent_unique[df_LSTM_sent_unique.LSTM_sent_cat==cat]
    word_list = ' '.join(df['tweet_text'].tolist())
    
    wordcloud_excstopwords = WordCloud(stopwords=stopwords_wc,
                                   background_color="white",
                                   max_words=len(word_list),max_font_size=40,
                                   relative_scaling=.5).generate(word_list)
    #plt.figure()
    ax[i].imshow(wordcloud_excstopwords)
    ax[i].axis("off")
    ax[i].set(title='LSTM sentiment: {0}'.format(cat))
fig.savefig('INDP/Images/LSTM_sent_wordclouds_sentcat.png')

# By UTLA
fig, ax = plt.subplots(6,4, figsize = (24,12))
for utla,ax_i in zip(utlas,ax.flatten()):
    df = df_LSTM_sent[df_LSTM_sent.area==utla]
    word_list = ' '.join(df['tweet_text'].tolist())
    
    wordcloud_excstopwords = WordCloud(stopwords=stopwords_wc,
                                   background_color="white",
                                   max_words=len(word_list),max_font_size=40,
                                   relative_scaling=.5).generate(word_list)
    #plt.figure()
    ax_i.imshow(wordcloud_excstopwords)
    ax_i.axis("off")
    ax_i.set(title='LSTM sentiment: {0}'.format(utla))
fig.savefig('INDP/Images/LSTM_sent_wordclouds_utla.png')

# By UTLA and sentiment category
for utla in utlas:
    fig, ax = plt.subplots(1,3, figsize = (16,12))
    for i,cat in enumerate(['Positive','Neutral','Negative']):
        df = df_LSTM_sent[df_LSTM_sent.LSTM_sent_cat==cat]
        word_list = ' '.join(df['tweet_text'].tolist())
        
        wordcloud_excstopwords = WordCloud(stopwords=stopwords_wc,
                                       background_color="white",
                                       max_words=len(word_list),
                                       max_font_size=40,
                                       relative_scaling=.5).generate(word_list)
        ax[i].imshow(wordcloud_excstopwords)
        ax[i].axis("off")
        ax[i].set(title='LSTM sentiment: {0}'.format(cat))
        fig.suptitle(utla, y=0.65)
        plt.tight_layout()
    fig.savefig('INDP/Images/LSTM_sent_wordclouds_sentcat_{0}.png'.format(utla))

#%% Compare sentiment scores over time

# Convert tweet_date to datetime
df_LSTM_sent_unique['tweet_date'] = pd.to_datetime(df_LSTM_sent_unique['tweet_date'])
# Split into initial data (Jan only - used for training) and recent data 
# (15 Feb onwards)
def date_split(var):
    if var <= pd.to_datetime('2022-01-31'):
        return 'Initial'
    elif var >= pd.to_datetime('2022-02-15'):
        return 'Recent'

df_LSTM_sent_unique['date_split'] = df_LSTM_sent_unique['tweet_date'].apply(date_split)
df_LSTM_sent_unique.groupby('date_split')['LSTM_sent'].mean()
df_LSTM_sent_unique.groupby(['date_split','LSTM_sent_cat']).size()/df_LSTM_sent_unique.groupby('date_split').size()

# Calculate rolling 7-day sentiment score
## Overall
df_7day = (df_LSTM_sent_unique.groupby('tweet_date')['LSTM_sent']
    .agg(['sum','count']).reset_index())

df_7day_rolling = df_7day.rolling(window=7).sum()
df_7day_rolling['mean'] = df_7day_rolling['sum']/df_7day_rolling['count']

df_7day_rolling = df_7day[['tweet_date']].join(df_7day_rolling)

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.lineplot(data=df_7day_rolling, x='tweet_date', y='mean', ax=ax, 
                   color='#007C91')
ax.set(xlabel = 'Tweet date', ylabel = '7-day rolling average sentiment score')
plt.tight_layout()
plt.savefig("INDP//Images//LSTM_sentiment.png")

## By search term


## By UTLA




# Plot overall 7-day average


# Plot 7-day average by search term


# Plot 7-day average by UTLA






