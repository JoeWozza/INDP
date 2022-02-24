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

df_LSTM_sent = pd.read_csv("INDP\Data\df_LSTM_sent.csv")
# Categorise sentiment score
df_LSTM_sent['LSTM_sent_cat'] = df_LSTM_sent.apply(lambda row: 
    class_lstm.cat_sentiment_str(row['LSTM_sent']), axis=1)
df_LSTM_sent_unique = df_LSTM_sent.drop_duplicates(subset=['tweet_id'])

#%%

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
