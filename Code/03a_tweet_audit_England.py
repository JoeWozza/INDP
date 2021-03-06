# -*- coding: utf-8 -*-
"""
This code is used to audit the England Tweets dataset.

@author: Joe.WozniczkaWells
"""

from os import chdir, getcwd,listdir

# Set file path
filepath = ("C:\\Users\\joew\\Documents\\Apprenticeship\\UoB\\SPFINDP21T4")
chdir(filepath)

from INDP.Code import CircleApprox
class_ca = CircleApprox.CircleApprox()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon
import math
from datetime import datetime
import os
import numpy as np

# Create folder in which to save visualisations
audit_folder = '{0}/INDP/Images/Tweet_audit/England'.format(filepath)

if not os.path.exists(audit_folder):
    os.makedirs(audit_folder)

#%%
files = listdir('INDP/Data/Tweets')

#Combine all England tweets
filestring_eng = 'df_tweets_eng_tweepy_'

tweets_files_eng = [s for s in files if filestring_eng in s]
tweets_files_eng.remove('df_tweets_eng_tweepy_2022-02-23.csv')

df_tweets_eng = pd.DataFrame()

for f in tweets_files_eng:
    print(f)
    df_tweets_eng = (df_tweets_eng
                     .append(pd.read_csv('INDP/Data/Tweets/{0}'.format(f))))

df_tweets_deduped = df_tweets_eng.drop_duplicates(subset=['tweet_id'])
# 86,444 tweets
df_tweets_deduped_term = (df_tweets_eng
                          .drop_duplicates(subset=['tweet_id','search_term']))
# 104,698 when deduplicated by search_term as well.

df_tweets_deduped.dtypes
# Convert tweet_datetime and tweet_date to datetime
df_tweets_deduped.tweet_datetime = (pd.to_datetime(
        df_tweets_deduped.tweet_datetime))
df_tweets_deduped.tweet_date = pd.to_datetime(df_tweets_deduped.tweet_date)
df_tweets_deduped['tweet_time'] = df_tweets_deduped.tweet_datetime.dt.time
df_tweets_deduped['tweet_hour'] = df_tweets_deduped.tweet_datetime.dt.hour
df_tweets_deduped.dtypes

#%%

# Plot frequency of tweets by date
tweets_date = (df_tweets_deduped
               .groupby(['tweet_date']).size().to_frame('tweets')
               .reset_index())

fig, ax = plt.subplots(figsize = (12,6))
# Have to use barplot rather than countplot to get dates in correct order
fig = sns.barplot(data=tweets_date, x='tweet_date', y='tweets', ax=ax, 
                  color='#007C91')
x_dates = np.datetime_as_string(tweets_date.tweet_date.sort_values().unique(), 
                                unit='D')
ax.set_xticklabels(x_dates,rotation = 90)
ax.set(xlabel='Date',ylabel='Number of Tweets')
plt.tight_layout()
plt.savefig('{0}/tweets_date.png'.format(audit_folder))

# Plot frequency by time of day
tweets_hour = (df_tweets_deduped
               .groupby(['tweet_hour']).size().to_frame('tweets')
               .reset_index())

fig, ax = plt.subplots(figsize = (12,6))
# Have to use barplot rather than countplot to get dates in correct order
fig = sns.barplot(data=tweets_hour, x='tweet_hour', y='tweets', ax=ax, 
                  color='#007C91')
ax.set(xlabel='Hour of day',ylabel='Number of Tweets')
plt.tight_layout()
plt.savefig('{0}/tweets_time.png'.format(audit_folder))

# Plot frequency by search term
tweets_search_term = (df_tweets_deduped_term
                      .groupby(['search_term']).size().to_frame('tweets')
                      .reset_index())
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.barplot(data=tweets_search_term, x='search_term', y='tweets', ax=ax, 
                  color='#007C91')
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
ax.set(xlabel='Search term',ylabel='Number of Tweets')
plt.tight_layout()
plt.savefig('{0}/tweets_search_term.png'.format(audit_folder))

# Frequency by location
tweets_user_location = (df_tweets_deduped[pd.notna(
        df_tweets_deduped.user_location)])

# location wordcloud
location_list = ' '.join(tweets_user_location['user_location'].tolist())

wordcloud_location = WordCloud(background_color="white",
                               max_words=len(location_list),max_font_size=40, 
                               relative_scaling=.5).generate(location_list)
plt.figure()
plt.imshow(wordcloud_location)
plt.axis("off")
plt.savefig('{0}/wordcloud_location.png'.format(audit_folder))

# Wordcloud with stop words excluded
stopwords=set(STOPWORDS)
# Exclude a few other meaningless words
stopwords.update(['England','United','Kingdom','UK'])

wordcloud_location = WordCloud(stopwords=stopwords,background_color="white",
                               max_words=len(location_list),max_font_size=40, 
                               relative_scaling=.5).generate(location_list)
plt.figure()
plt.imshow(wordcloud_location)
plt.axis("off")
plt.savefig('{0}/wordcloud_location_excstopwords.png'.format(audit_folder))

# Frequency by user
tweets_user_name = (df_tweets_deduped_term
                    .groupby(['user_name']).size().to_frame('tweets')
                    .reset_index())

# username wordcloud
username_list = ' '.join(df_tweets_deduped['user_name'].tolist())

wordcloud_username = (WordCloud(background_color="white",
                                max_words=len(username_list),max_font_size=40, 
                                relative_scaling=.5,
                                collocations=False).generate(username_list))
plt.figure()
plt.imshow(wordcloud_username)
plt.axis("off")
plt.savefig('{0}/wordcloud_username.png'.format(audit_folder))

# tweet_text wordclouds
word_list = ' '.join(df_tweets_deduped['tweet_text'].tolist())

wordcloud = WordCloud(background_color="white",
                      max_words=len(word_list),max_font_size=40, 
                      relative_scaling=.5).generate(word_list)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('{0}/wordcloud_all.png'.format(audit_folder))

# Wordcloud with stop words excluded
stopwords=set(STOPWORDS)
# Exclude a few other meaningless words
stopwords.update(['https','co','t','s','amp','u'])

wordcloud_excstopwords = WordCloud(stopwords=stopwords,
                                   background_color="white",
                                   max_words=len(word_list),max_font_size=40,
                                   relative_scaling=.5).generate(word_list)
plt.figure()
plt.imshow(wordcloud_excstopwords)
plt.axis("off")
plt.savefig('{0}/wordcloud_excstopwords.png'.format(audit_folder))

# Wordclouds by search term
for term in (df_tweets_deduped_term
             .drop_duplicates(['search_term'])['search_term'].tolist()):
    print(term)
    df = df_tweets_deduped_term[df_tweets_deduped_term['search_term']==term]
    word_list = ' '.join(df['tweet_text'].tolist())
    
    wordcloud = WordCloud(background_color="white",
                      max_words=len(word_list),max_font_size=40, 
                      relative_scaling=.5).generate(word_list)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('{0}/wordcloud_all_{1}.png'.format(audit_folder,term))
    
    wordcloud_excstopwords = WordCloud(stopwords=stopwords,
                                   background_color="white",
                                   max_words=len(word_list),max_font_size=40,
                                   relative_scaling=.5).generate(word_list)
    plt.figure()
    plt.imshow(wordcloud_excstopwords)
    plt.axis("off")
    plt.savefig('{0}/wordcloud_excstopwords_{1}.png'.format(audit_folder,term))

# User bio wordcloud
tweets_user_bio = df_tweets_deduped[pd.notna(df_tweets_deduped.user_bio)]
word_list = ' '.join(tweets_user_bio['user_bio'].tolist())

wordcloud = WordCloud(stopwords=stopwords,background_color="white",
                      max_words=len(word_list),max_font_size=40, 
                      relative_scaling=.5).generate(word_list)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('{0}/wordcloud_user_bio.png'.format(audit_folder))