# -*- coding: utf-8 -*-
"""
This code cleans the data downloaded from Twitter.

@author: Joe Wozniczka-Wells
"""

from os import chdir, getcwd,listdir

# Set file path
filepath = ("C:\\Users\\joew\\Documents\\Apprenticeship\\UoB\\SPFINDP21T4\\")
chdir(filepath)

from INDP.Code import VADER
class_v = VADER.VADER()

# Load packages
import pandas as pd
import os

tweet_folder = 'INDP/Data/Tweets'

#%% Read in data

# Define common string at start of file names
filestring_eng = 'df_tweets_eng_tweepy_'
filestring_mids = 'df_tweets_tweepy_'
# Get list of all files in path
files = listdir(tweet_folder)

# Get lists of all files containing common strings
tweets_files_eng = [s for s in files if filestring_eng in s]
tweets_files_mids = [s for s in files if filestring_mids in s]
# Combine data into single dataframe
df_tweets_eng = pd.DataFrame()
for f in tweets_files_eng:
    print(f)
    df_tweets_eng = df_tweets_eng.append(pd.read_csv("{0}/{1}".format(
            tweet_folder,f)))
    
df_tweets_mids = pd.DataFrame()
for f in tweets_files_mids:
    print(f)
    df_tweets_mids = df_tweets_mids.append(pd.read_csv("{0}/{1}".format(
            tweet_folder,f)))

#%% Start cleaning data

# Remove twitter handles, URLs, special punctuation and numbers
df_tweets_eng['tweet_text'] = df_tweets_eng['tweet_text'].apply(class_v.clean)
df_tweets_mids['tweet_text'] = (df_tweets_mids['tweet_text']
    .apply(class_v.clean))

# Remove Tweets that contain nothing else
df_tweets_eng = (df_tweets_eng[
        df_tweets_eng.tweet_text.str.strip().str.len() > 0].reset_index())
df_tweets_mids = (df_tweets_mids[
        df_tweets_mids.tweet_text.str.strip().str.len() > 0].reset_index())
    
# Remove Tweets with location 'Gotham City'. These have been erroneously placed
# in Nottinghamshire due to Twitter confusing Gotham City for Gotham in Notts
df_tweets_eng = df_tweets_eng[df_tweets_eng.user_location.str
                              .contains('Gotham City',na=False) == False]
df_tweets_mids = df_tweets_mids[df_tweets_mids.user_location.str
                                .contains('Gotham City',na=False) == False]

df_tweets_eng.to_csv('{0}/df_tweets_eng.csv'.format(tweet_folder))
df_tweets_mids.to_csv('{0}/df_tweets_mids.csv'.format(tweet_folder))

