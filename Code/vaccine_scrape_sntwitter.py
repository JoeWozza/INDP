# -*- coding: utf-8 -*-
"""
This script uses the TweetScrape class from sntwitterScrape.py to download 
vaccine-related Tweets from the duration of the pandemic, from 11/3/22 to the 
day the code is executed.

@author: Joe Wozniczka-Wells
"""

from os import chdir

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

from INDP.Code import sntwitterScrape
class_ts = sntwitterScrape.TweetScrape()

import pandas as pd
import snscrape.modules.twitter as sntwitter
import os

from datetime import datetime
from time import sleep
        
# Create folders in which to save twitter data
data_folder = '{0}/INDP/Data'.format(filepath)
tweets_folder = '{0}/INDP/Data/Tweets'.format(filepath)

if not os.path.exists(data_folder):
    os.makedirs(data_folder)
if not os.path.exists(tweets_folder):
    os.makedirs(tweets_folder)

# Read in df_utlas from csv
df_utlas = pd.read_csv("INDP/Geo/df_utlas_90_95.csv")

searchTerms = ['vaccines','vaccine','vaccinated',
               'vaccination','booster','pfizer',
               'vaccinations','unvaccinated',
               'astrazenica','antivaxxers',
               'vaccinate','vax','vaxxed']
#%% Functions

# Create empty dataframes
df_tweets = pd.DataFrame()
df_error = pd.DataFrame()
# Set since and until for entire period
since = '2020-03-11'
until = datetime.now().strftime("%Y-%m-%d")

df_tweets, df_error = class_ts.sntwitter_scrape_areas(df_tweets,df_error,
                                                      df_utlas,'utla',since,
                                                      until,searchTerms)

# Remove tweets that do not contain any of the search terms 
# (sometimes sntwitter downloads tweets where the search term is in 
# the user's bio)
df_tweets = (df_tweets[df_tweets['tweet_text'].str.lower().str
                       .contains('|'.join(searchTerms))])

df_tweets_deduped = df_tweets.drop_duplicates(subset=['tweet_id','area'])

# Output to csv
df_tweets.to_csv('{0}/df_tweets_sntwitter_{1}.csv'.format(tweets_folder,
                 str(datetime.now().date())))
df_tweets_deduped.to_csv('{0}/df_tweets_deduped_sntwitter_{1}.csv'.format(
        tweets_folder,str(datetime.now().date())))