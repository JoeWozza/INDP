# -*- coding: utf-8 -*-
"""
This script uses the TweetScrape class from TweepyScrape.py to download 
vaccine-related Tweets from the last 9 days.

This script was run periodically from 9/1/22 to 6/2/22, to collect a sufficient
number of Tweets over a reasonable period of time, on which to train the model.

@author: Joe Wozniczka-Wells
"""
#%% Packages

from os import chdir, getcwd

# Set file path
filepath = ("C:\\Users\\joew\\Documents\\Apprenticeship\\UoB\\SPFINDP21T4\\")
chdir(filepath)

from INDP.Code import TweepyScrape
class_ts = TweepyScrape.TweetScrape()

import pandas as pd
import tweepy
from datetime import datetime
import os

# Create folders in which to save twitter data
data_folder = '{0}/INDP/Data'.format(filepath)
tweets_folder = '{0}/INDP/Data/Tweets'.format(filepath)

if not os.path.exists(data_folder):
    os.makedirs(data_folder)
if not os.path.exists(tweets_folder):
    os.makedirs(tweets_folder)

#%% Twitter API access stuff

# Keys from here: https://developer.twitter.com/en/portal/projects/
# 1475494865567899649/apps/new
consumer_key = ''
consumer_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit = True)

#%% Data retrieval

# Read in df_areas from csv (for now, may do this all within Python eventually)
df_utlas = pd.read_csv("INDP/Geo/df_utlas_90_95.csv")

searchTerms = ['vaccines','vaccine','vaccinated',
               'vaccination','booster','pfizer',
               'vaccinations','unvaccinated',
               'astrazenica','antivaxxers',
               'vaccinate','vax','vaxxed']

# Initiate dataframes to collect tweets and record timings
df_tweets = pd.DataFrame()
df_timings = pd.DataFrame()

start = datetime.now()
# Loop through UTLAs
utlas = df_utlas.drop_duplicates(subset=['utla']).utla

df_tweets, df_timings = class_ts.circles_scrape_areas(api,utlas,df_utlas,
                                                      'utla',df_tweets,
                                                      df_timings,searchTerms)

# Whole Midlands - download Tweets from Midlands and use names of UTLAs to
# assign UTLAs where possible.
# Initiate dataframe to collect tweets and record timings
df_mids_tweets = pd.DataFrame()
df_mids_timings = pd.DataFrame()

# Download tweets    
df_mids_tweets, df_mids_timings = class_ts.tweepy_scrape_terms(api,
                                                               df_mids_tweets,
                                                               df_mids_timings,
                                                               searchTerms,
                                                               52.8052096,
                                                               -1.3846729,150,
                                                               'Midlands',
                                                               'Midlands')

# Create date field from datetime
df_mids_tweets['tweet_date'] = df_mids_tweets.tweet_datetime.dt.date

end = datetime.now()
print('Overall: ' + str(end-start))

# Distribute Midlands circle tweets to UTLAs using user_location
df_tweets = df_tweets.append(class_ts.manual_assign_areas(df_mids_tweets,
                                                          utlas))

# Deduplicate by tweet_id and utla
df_tweets_deduped = df_tweets.drop_duplicates(subset=['tweet_id','area'])

# UTLA frequencies
print(pd.value_counts(df_tweets.area))
print(pd.value_counts(df_tweets_deduped.area))
print(pd.value_counts(df_tweets_deduped.tweet_date))

# Output to csvs
df_tweets.to_csv('{0}/df_tweets_tweepy_{1}.csv'.format(tweets_folder,
                 str(datetime.now().date())))
df_tweets_deduped.to_csv('{0}/df_tweets_deduped_tweepy_{1}.csv'.format(
    tweets_folder,str(datetime.now().date())))
df_mids_tweets.to_csv('{0}/df_mids_tweets_tweepy_{1}.csv'.format(tweets_folder,
                      str(datetime.now().date())))

#%%
# Tweets from all of England
df_eng = pd.read_csv("INDP/Geo/df_eng_90_95.csv")

# Initiate dataframes to collect tweets and record timings
df_tweets_eng = pd.DataFrame()
df_timings_eng = pd.DataFrame()

start = datetime.now()

df_tweets_eng, df_timings_eng = class_ts.circles_scrape(api,"England",df_eng,
                                                        df_tweets_eng,
                                                        df_timings_eng,
                                                        searchTerms)

end = datetime.now()
print('Overall: ' + str(end-start))

df_tweets_eng_deduped = df_tweets_eng.drop_duplicates(subset=['tweet_id'])

# Output to csvs
df_tweets_eng.to_csv('{0}/df_tweets_eng_tweepy_{1}.csv'.format(tweets_folder,
                     str(datetime.now().date())))
df_tweets_eng_deduped.to_csv('{0}/df_tweets_eng_deduped_tweepy_{1}.csv'.format(
        tweets_folder, str(datetime.now().date())))
