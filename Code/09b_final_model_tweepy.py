# -*- coding: utf-8 -*-
"""
This code is used to investigate the output from the final model on the Tweets
downloaded using TweePy. This includes producing visualisations for the final
report.

@author: Joe Wozniczka-Wells
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import os
from itertools import product

from os import chdir

# Set file path
filepath = ("C:\\Users\\joew\\Documents\\Apprenticeship\\UoB\\SPFINDP21T4\\")
chdir(filepath)

from INDP.Code import LSTM
class_lstm = LSTM.LSTM()
from INDP.Code import VADER
class_v = VADER.VADER()

# Create folders
images_folder = 'INDP/Images'
finalmodel_folder = 'INDP/Images/Final_model'

if not os.path.exists(images_folder):
    os.makedirs(images_folder)
if not os.path.exists(finalmodel_folder):
    os.makedirs(finalmodel_folder)

df_tweepy_LSTM_sent = pd.read_csv("INDP/Data/LSTM/df_tweepy_LSTM_sent.csv")
# Categorise sentiment score
df_tweepy_LSTM_sent['LSTM_sent_cat'] = df_tweepy_LSTM_sent.apply(lambda row: 
    class_lstm.cat_sentiment_str(row['LSTM_sent']), axis=1)
# Convert tweet_date to datetime
df_tweepy_LSTM_sent['tweet_date'] = pd.to_datetime(
        df_tweepy_LSTM_sent['tweet_date'])
df_tweepy_LSTM_sent_unique = df_tweepy_LSTM_sent.drop_duplicates(
        subset=['tweet_id'])

search_terms = (df_tweepy_LSTM_sent_unique.drop_duplicates(
        subset=['search_term'])['search_term'].sort_values())
utlas = (df_tweepy_LSTM_sent.drop_duplicates(subset=['area'])
    ['area'].sort_values())

#%% Investigate sentiment scores, overall, by search term and by UTLA

# Plot distribution of sentiment scores
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.distplot(df_tweepy_LSTM_sent_unique['LSTM_sent'], ax=ax,
                   color='#007C91', kde_kws={'clip': (-1,1)})                
ax.set(xlabel = 'LSTM sentiment score')
plt.tight_layout()
plt.savefig("{0}/LSTM_tweepy_sentiment.png".format(finalmodel_folder))

# Plot distribution of sentiment scores by search term
g = sns.FacetGrid(df_tweepy_LSTM_sent_unique, col = 'search_term', 
                  col_wrap = 5, col_order = search_terms)
g.map(sns.histplot, 'LSTM_sent', color='#007C91', kde=True, 
      kde_kws={'clip': (-1,1)}, linewidth=0)
g.set_axis_labels(x_var = 'LSTM sentiment score', y_var = 'Density')
g.set_titles(col_template = '{col_name}')
g.tight_layout()
g.savefig("{0}/LSTM_tweepy_sentiment_search_term_hist.png".format(
        finalmodel_folder))

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.boxplot(data=df_tweepy_LSTM_sent_unique, x='search_term', 
                  y='LSTM_sent', color='#007C91', order = search_terms)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.savefig("{0}/LSTM_tweepy_sentiment_search_term_box.png".format(
        finalmodel_folder))

# Plot distribution of sentiment scores by UTLA
g = sns.FacetGrid(df_tweepy_LSTM_sent, col = 'area', col_wrap = 5,
                  col_order = utlas)
g.map(sns.histplot, 'LSTM_sent', color='#007C91', kde=True, 
      kde_kws={'clip': (-1,1)}, linewidth=0)
g.set_axis_labels(x_var = 'LSTM sentiment score', y_var = 'Density')
g.set_titles(col_template = '{col_name}')
g.tight_layout()
g.savefig("{0}/LSTM_tweepy_sentiment_UTLA_hist.png".format(finalmodel_folder))

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.boxplot(data=df_tweepy_LSTM_sent, x='area', y='LSTM_sent', 
                  color='#007C91', order = utlas)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.savefig("{0}/LSTM_tweepy_sentiment_UTLA_box.png".format(finalmodel_folder))

#%% Word clouds
# Wordcloud with stop words excluded
stopwords_wc=set(STOPWORDS)
# Exclude a few other meaningless words
stopwords_wc.update(['https','co','t','s','amp','u','covid'])
stopwords_wc.update(search_terms)

# By sentiment category
fig, ax = plt.subplots(1,3, figsize = (24,12))
for i,cat in enumerate(['Positive','Neutral','Negative']):
    df = df_tweepy_LSTM_sent_unique[df_tweepy_LSTM_sent_unique.
                                    LSTM_sent_cat==cat]
    word_list = ' '.join(df['tweet_text'].tolist())
    
    wordcloud_excstopwords = WordCloud(stopwords=stopwords_wc,
                                   background_color="white",
                                   max_words=len(word_list),max_font_size=40,
                                   relative_scaling=.5).generate(word_list)
    ax[i].imshow(wordcloud_excstopwords)
    ax[i].axis("off")
    ax[i].set(title='LSTM sentiment: {0}'.format(cat))
fig.savefig('{0}/LSTM_tweepy_sent_wordclouds_sentcat.png'.format(
        finalmodel_folder))

# By UTLA
fig, ax = plt.subplots(6,4, figsize = (24,12))
for utla,ax_i in zip(utlas,ax.flatten()):
    df = df_tweepy_LSTM_sent[df_tweepy_LSTM_sent.area==utla]
    word_list = ' '.join(df['tweet_text'].tolist())
    
    wordcloud_excstopwords = WordCloud(stopwords=stopwords_wc,
                                   background_color="white",
                                   max_words=len(word_list),max_font_size=40,
                                   relative_scaling=.5).generate(word_list)
    #plt.figure()
    ax_i.imshow(wordcloud_excstopwords)
    ax_i.axis("off")
    ax_i.set(title='LSTM sentiment: {0}'.format(utla))
fig.savefig('{0}/LSTM_tweepy_sent_wordclouds_utla.png'.format(
        finalmodel_folder))

# By UTLA and sentiment category
for utla in utlas:
    fig, ax = plt.subplots(1,3, figsize = (16,12))
    for i,cat in enumerate(['Positive','Neutral','Negative']):
        df = df_tweepy_LSTM_sent[df_tweepy_LSTM_sent.LSTM_sent_cat==cat]
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
    fig.savefig('{0}/LSTM_tweepy_sent_wordclouds_sentcat_{1}.png'.format(
            finalmodel_folder,utla))

#%% Compare sentiment scores over time

#%% 7-day rolling averages

# Calculate rolling 7-day sentiment score
## Overall
df_tweepy_7day = (df_tweepy_LSTM_sent_unique.groupby('tweet_date')['LSTM_sent']
    .agg(['sum','count']).asfreq('d').reset_index())
df_tweepy_7day['sum'] = df_tweepy_7day['sum'].fillna(0)
df_tweepy_7day['count'] = df_tweepy_7day['count'].fillna(0)

df_tweepy_7day_rolling = df_tweepy_7day.rolling(window=7).sum()
df_tweepy_7day_rolling['mean'] = (df_tweepy_7day_rolling['sum']/
                      df_tweepy_7day_rolling['count'])
df_tweepy_7day_rolling['mean'] = df_tweepy_7day_rolling['mean'].fillna(np.inf)

df_tweepy_7day_rolling = df_tweepy_7day[['tweet_date']].join(
        df_tweepy_7day_rolling)

all_dates = df_tweepy_7day.tweet_date
## By search term
dates_searchterms = pd.DataFrame(list(product(all_dates,search_terms)),
                                 columns=['tweet_date','search_term'])

df_tweepy_7day_search_term = (df_tweepy_LSTM_sent_unique.groupby(
        ['tweet_date','search_term'])['LSTM_sent']
    .agg(['sum','count']).reset_index().merge(dates_searchterms, 
        on=['tweet_date','search_term'], how='right'))
df_tweepy_7day_search_term['sum'] = df_tweepy_7day_search_term['sum'].fillna(0)
df_tweepy_7day_search_term['count'] = (
        df_tweepy_7day_search_term['count'].fillna(0))

df_tweepy_7day_search_term['sum'] = (df_tweepy_7day_search_term
                          .groupby(['search_term'])['sum']
                          .transform(lambda x: x.rolling(7).sum()))
df_tweepy_7day_search_term['count'] = (df_tweepy_7day_search_term
                          .groupby(['search_term'])['count']
                          .transform(lambda x: x.rolling(7).sum()))

df_tweepy_7day_search_term['mean'] = (df_tweepy_7day_search_term['sum']/
                          df_tweepy_7day_search_term['count'])
df_tweepy_7day_search_term['mean'] = (
        df_tweepy_7day_search_term['mean'].fillna(np.inf))

## By UTLA
dates_tweepy_utlas = pd.DataFrame(list(product(all_dates,utlas)),
                                 columns=['tweet_date','area'])

df_tweepy_7day_utla = (df_tweepy_LSTM_sent
                       .groupby(['tweet_date','area'])['LSTM_sent']
                       .agg(['sum','count']).reset_index()
                       .merge(dates_tweepy_utlas, on=['tweet_date','area'], 
                              how='right'))
df_tweepy_7day_utla['sum'] = df_tweepy_7day_utla['sum'].fillna(0)
df_tweepy_7day_utla['count'] = df_tweepy_7day_utla['count'].fillna(0)

df_tweepy_7day_utla['sum'] = (df_tweepy_7day_utla
                   .groupby(['area'])['sum']
                   .transform(lambda x: x.rolling(7).sum()))
df_tweepy_7day_utla['count'] = (df_tweepy_7day_utla
                   .groupby(['area'])['count']
                   .transform(lambda x: x.rolling(7).sum()))

df_tweepy_7day_utla['mean'] = (df_tweepy_7day_utla['sum']/
                   df_tweepy_7day_utla['count'])
df_tweepy_7day_utla['mean'] = df_tweepy_7day_utla['mean'].fillna(np.inf)

# Plot overall 7-day average
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.lineplot(data=df_tweepy_7day_rolling,x='tweet_date',y='mean',ax=ax, 
                   color='#007C91')
ax.set(xlabel = 'Tweet date', ylabel = '7-day rolling average sentiment score')
plt.tight_layout()
plt.savefig("{0}/LSTM_tweepy_sentiment_roll7.png".format(finalmodel_folder))

# Plot 7-day average by search term
g = sns.relplot(data=df_tweepy_7day_search_term, x='tweet_date', y='mean', 
                col='search_term', col_wrap=5, kind='line', color='#007C91')
g.set_axis_labels(x_var = 'Tweet date', 
                  y_var = 'Mean sentiment score')
g.set_titles(col_template = 'Search term: {col_name}')
g.fig.suptitle('Sentiment scores over time, by search term')
g.tight_layout()
g.savefig("{0}/LSTM_tweepy_sentiment_roll7_searchterms.png".format(
        finalmodel_folder))

# Plot 7-day average by UTLA
g = sns.relplot(data=df_tweepy_7day_utla, x='tweet_date', y='mean', 
                col='area', col_wrap=5, kind='line', color='#007C91')
g.set_axis_labels(x_var = 'Tweet date', 
                  y_var = 'Mean sentiment score')
g.set_titles(col_template = '{col_name}')
g.fig.suptitle('Sentiment scores over time, by UTLA')
g.tight_layout()
g.savefig("{0}/LSTM_tweepy_sentiment_roll7_UTLAs.png".format(
        finalmodel_folder))

#%% Jan/Feb split

# Split into initial data (Jan only - used for training) and recent data 
# (15 Feb onwards)
def date_split(var):
    if var <= pd.to_datetime('2022-01-31'):
        return 'Initial'
    elif var >= pd.to_datetime('2022-02-15'):
        return 'Recent'

df_tweepy_LSTM_sent_unique['date_split'] = (
        df_tweepy_LSTM_sent_unique['tweet_date'].apply(date_split))
df_tweepy_LSTM_sent_unique['month_year'] = (
        pd.to_datetime(df_tweepy_LSTM_sent_unique['tweet_date'])
        .dt.to_period('M'))

print(df_tweepy_LSTM_sent_unique.groupby('month_year')['LSTM_sent'].mean())
print(df_tweepy_LSTM_sent_unique
      .groupby(['month_year','LSTM_sent_cat']).size()/
      df_tweepy_LSTM_sent_unique.groupby('month_year').size())

df_tweepy_LSTM_sent_unique_JanFeb = (
        df_tweepy_LSTM_sent_unique[pd.to_datetime(
                df_tweepy_LSTM_sent_unique.tweet_date)>='2022-01-01'])

df_tweepy_LSTM_sent['month_year'] = (
        pd.to_datetime(df_tweepy_LSTM_sent['tweet_date']).dt.to_period('M'))
df_tweepy_LSTM_sent_JanFeb = (
        df_tweepy_LSTM_sent[pd.to_datetime(
                df_tweepy_LSTM_sent.tweet_date)>='2022-01-01'])

# Overall box plot
fig, ax = plt.subplots(figsize=(12,6))
fig = sns.boxplot(data=df_tweepy_LSTM_sent_unique_JanFeb, x='month_year', 
                  y='LSTM_sent', color='#007C91')
ax.set(xlabel = 'Tweet date', ylabel = 'Sentiment score', 
       title = 'Sentiment scores: January 2022 v February 2022')
plt.tight_layout()
plt.savefig("{0}/LSTM_tweepy_sentiment_JanFeb_box.png".format(
        finalmodel_folder))

# UTLA box plot
g = sns.catplot(data=df_tweepy_LSTM_sent_JanFeb, x='month_year', y='LSTM_sent', 
                col='area', col_wrap=5, kind='box', color='#007C91')
g.set_axis_labels(x_var = 'Tweet date', 
                  y_var = 'Sentiment score')
g.set_titles(col_template = '{col_name}')
g.fig.suptitle('Sentiment scores: January 2022 v February 2022, by UTLA')
g.tight_layout()
g.savefig("{0}/LSTM_tweepy_sentiment_JanFeb_UTLAs_box.png".format(
        finalmodel_folder))
