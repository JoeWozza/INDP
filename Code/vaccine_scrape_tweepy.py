# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:01:14 2022

@author: Joe.WozniczkaWells
"""
#%% Packages

import pandas as pd
import tweepy
from datetime import datetime

#%% Twitter API access stuff

# Keys from here: https://developer.twitter.com/en/portal/projects/1475494865567899649/apps/new
consumer_key = 'OPqZPaFbJHx8sXf8y7C5umylY'
consumer_secret = 'qE6FdosS3aiSHNDcamEhxcGCxl40k4oKoPNoVTrrW8IMWkYSkB'
# See here for keeping them secure: https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit = True)

#%% Data retrieval

# Read in df_utlas from csv (for now, will do this all within Python eventually)
df_utlas = pd.read_csv("df_utlas_90.csv")
df_utlas = df_utlas[df_utlas.utla.isin(['Derbyshire','Derby','Nottingham','Nottinghamshire'])]

searchTerms = ['vaccines','vaccine','vaccinated',
               'vaccination','booster','pfizer',
               'vaccinations','unvaccinated',
               'astrazenica','antivaxxers',
               'vaccinate','vax','vaxxed']

#


# Initiate dataframe to collect tweets
df_tweets = pd.DataFrame(columns = ['utla','utla_circle','search_term','tweet_id','tweet_url','tweet_text',
                                    'tweet_datetime','tweet_place','tweet_coords',
                                    'tweet_language','tweet_likes',
                                    'tweet_retweets','user_name',
                                    'user_location','user_followers',
                                    'user_protected','user_tweets','user_likes',
                                    'user_bio'])

# Initiate dataframe to record timings
df_timings = pd.DataFrame(columns = ['utla','utla_circle','search_term','no_tweets','time_taken'])

start = datetime.now()
# Loop through UTLAs
for utla in df_utlas.drop_duplicates(subset=['utla']).utla:
    
    # Loop through circles
    df_utla = df_utlas[df_utlas.utla == utla]#.head(5)
    
    for c,(lat,long,radius) in enumerate(zip(df_utla.lat, df_utla.long, 
          df_utla.radius)):
        
        # Loop through search terms
        for term in searchTerms:
            #lat = 52.914639
            #long = -1.47189
            #radius = 4.0
            #utla = 'Derby'
            
            print(utla + ': circle ' + str(c+1) + ' of ' + str(len(df_utla)) + ' - ' + term)
            
            term_start = datetime.now()
            # Collect tweets in df_tweets
            tweets = 0
            for tweet in tweepy.Cursor(api.search_tweets, q=term, 
                                       geocode='{0},{1},{2}km'.format(lat,long,radius),
                                       #until='2022-01-07',
                                       tweet_mode='extended').items(999999999):
                #print(tweet)
                #print(tweet.created_at)
                #print(tweet.id)
                #print(tweet.full_text)
                #print(tweet.place)
                #print(tweet.coordinates)
                #print(tweet.user.name)
                #print(tweet.user.location)
                #print('')
                dict_tweet = {'utla': utla, 'utla_circle': c+1, 'search_term': term,
                              'tweet_url': 'twitter.com/'+tweet.user.screen_name+'/status/'
                                  +tweet.id_str, #so I can find the tweet
                              'tweet_id': tweet.id, #so I can find the tweet
                              'tweet_text': tweet.full_text, #to analyse for sentiment
                              'tweet_datetime': tweet.created_at, #time and data
                              'tweet_place': tweet.place, #usually blank, but could be used if available
                              'tweet_coords': tweet.coordinates, #usually blank, but could be used if available
                              'tweet_language': tweet.lang, #filter to use 'en' only
                              'tweet_likes': tweet.favorite_count, #number of times liked - can be used to measure impact and possibly weight
                              'tweet_retweets': tweet.retweet_count, #number times retweeted - can be used to measure impact and possibly weight
                              'user_name': tweet.user.screen_name, #so I can find the user
                              'user_location': tweet.user.location, #location of user
                              'user_followers': tweet.user.followers_count, #number of followers - proxy for 'influence'
                              'user_protected': tweet.user.protected, #whether user is protected, expected to be false for all
                              'user_tweets': tweet.user.statuses_count, #number of tweets posted by user - proxy for 'activity'
                              'user_likes': tweet.user.favourites_count, #number of tweets liked by user - proxy for 'activity'
                              'user_bio': tweet.user.description} #user bio - could be searched for common pro-/anti-vaccination content
                df_tweets = df_tweets.append(dict_tweet, ignore_index=True)
                tweets += 1
            
            #Record timings
            term_end = datetime.now()
            time_taken = term_end - term_start
            dict_timings = {'utla': utla, 'utla_circle': c+1, 
                            'search_term': term, 'no_tweets': tweets,
                            'time_taken': str(time_taken)}
            df_timings = df_timings.append(dict_timings, ignore_index=True)
            print(str(time_taken))

end = datetime.now()
print(str(end-start))
# Twitter API only returns data for the previous 7 days

