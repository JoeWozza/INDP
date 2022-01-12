# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:01:14 2022

@author: Joe.WozniczkaWells
"""
#%% Packages

import pandas as pd
import tweepy
from datetime import datetime

# Set file path and create log file for task scheduler
filepath = "C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\SPFINDP21T4\\"

#%% Twitter API access stuff

# Keys from here: https://developer.twitter.com/en/portal/projects/
# 1475494865567899649/apps/new
consumer_key = 'OPqZPaFbJHx8sXf8y7C5umylY'
consumer_secret = 'qE6FdosS3aiSHNDcamEhxcGCxl40k4oKoPNoVTrrW8IMWkYSkB'
# See here for keeping them secure: https://developer.twitter.com/en/docs
# /twitter-api/getting-started/getting-access-to-the-twitter-api

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit = True)

#%% Define functions

def tweepy_scrape(api,df,df_timings,term,lat,long,radius,area,area_circle):
    # Uses tweepy's api.search_tweets function to scrape tweets and add them to
    # an existing dataframe, df. The function outputs df and a separate
    # dataframe, df_timings, that captures information on the volumes of tweets 
    # downloaded and time taken.
    # api: valid tweepy.api.API object
    # df: name of dataframe to contain tweets
    # df_timings: name of dataframe to contain timings
    # term: search term to be used to find tweets
    # lat: latitude of circle centroid to be used to find tweets
    # long: longitude of circle centroid to be used to find tweets
    # radius: radius of circle centroid to be used to find tweets
    # radius: radius of circle centroid to be used to find tweets
    # area: name of geographical area
    # area_circle: name/number of geographical area circle
    
    print(area + ': circle ' + area_circle + ' - ' + term)
    
    term_start = datetime.now()
    # Collect tweets in df_tweets
    tweets = 0
    for tweet in tweepy.Cursor(api.search_tweets, q=term, 
                               geocode='{0},{1},{2}km'.format(lat,long,radius),
                               tweet_mode='extended').items(999999999):
        dict_tweet = {'area': area, 'area_circle': area_circle, 
                      'search_term': term,
                      'tweet_url': 'twitter.com/'+tweet.user.screen_name+
                          '/status/'+tweet.id_str,
                      'tweet_id': tweet.id,
                      'tweet_text': tweet.full_text,
                      'tweet_datetime': tweet.created_at,
                      'tweet_place': tweet.place,
                      'tweet_coords': tweet.coordinates,
                      'tweet_language': tweet.lang,
                      'tweet_likes': tweet.favorite_count,
                      'tweet_retweets': tweet.retweet_count,
                      'user_name': tweet.user.screen_name,
                      'user_location': tweet.user.location,
                      'user_followers': tweet.user.followers_count,
                      'user_protected': tweet.user.protected,
                      'user_tweets': tweet.user.statuses_count,
                      'user_likes': tweet.user.favourites_count,
                      'user_bio': tweet.user.description}
        df = df.append(dict_tweet, ignore_index=True)
        tweets += 1
    
    #Record timings
    term_end = datetime.now()
    time_taken = term_end - term_start
    dict_timings = {'area': area, 'area_circle': area_circle, 
                    'search_term': term, 'no_tweets': tweets,
                    'time_taken': str(time_taken)}
    df_timings = df_timings.append(dict_timings, ignore_index=True)
    print(str(time_taken))
    
    return df, df_timings

def check_circle(check_term,lat,long,radius):
    # Checks whether the geographical circle contains any tweets containing a
    # user-defined search term.
    circ_tweets = 0
    for tweet in tweepy.Cursor(api.search_tweets, q = check_term, 
                               geocode='{0},{1},{2}km'.format(lat,long,
                                        radius)
                               ).items(1):
        circ_tweets += 1
        if circ_tweets >= 1:
            circ_tweets = True
        else:
            circ_tweets = False
    return circ_tweets

def circles_scrape(df_areas,area_col,df_tweets,df_timings):
    # Loop through circles
    df_area = df_areas[df_areas[area_col] == area]#.head(5)
    
    for c,(lat,long,radius) in enumerate(zip(df_area.lat, df_area.long, 
          df_area.radius)):
        
        # Are there any tweets from the circle? Test using search term 
        # 'twitter'.
        circ_tweets = check_circle('twitter',lat,long,radius)
        
        # If there are no tweets in the circle, skip it, otherwise search for
        # tweets.
        if circ_tweets == False:
            print('Skip circle ' + str(c+1) + ' in ' + area)
        else:
            # Loop through search terms
            for term in searchTerms:
                df_tweets, df_timings = tweepy_scrape(api,df_tweets,df_timings,
                                                      term,lat,long,radius,
                                                      area,str(c+1))

    # Create date field from datetime
    df_tweets['tweet_date'] = df_tweets.tweet_datetime.dt.date
    
    return df_tweets, df_timings

def manual_assign_areas(df_in,df_tweets):
    df_temp = df_in[df_in.user_location.str.contains(area)]
    df_temp.area = area
    df_tweets = df_tweets.append(df_temp)
    
    return df_tweets
    

#%% Data retrieval

# Read in df_areas from csv (for now, may do this all within Python eventually)
df_utlas = pd.read_csv("{0}df_utlas_90.csv".format(filepath))
df_utlas = df_utlas[df_utlas.utla.isin(['Leicestershire','Derby'])]

searchTerms = ['vaccines','vaccine','vaccinated',
               'vaccination','booster','pfizer',
               'vaccinations','unvaccinated',
               'astrazenica','antivaxxers',
               'vaccinate','vax','vaxxed']
searchTerms = ['vaccines']

# Initiate dataframes to collect tweets and record timings
df_tweets = pd.DataFrame()
df_timings = pd.DataFrame()

start = datetime.now()
# Loop through UTLAs
for area in df_utlas.drop_duplicates(subset=['utla']).utla:
    
    df_tweets, df_timings = circles_scrape(df_utlas,'utla',df_tweets,
                                           df_timings)

# Whole Midlands
# Initiate dataframe to collect tweets and record timings
df_mids_tweets = pd.DataFrame()
df_mids_timings = pd.DataFrame()

# Download tweets    
for term in searchTerms:
    df_mids_tweets, df_mids_timings = tweepy_scrape(api,df_mids_tweets,
                                                    df_mids_timings,term,
                                                    52.8052096,-1.3846729,150,
                                                    'Midlands','Midlands')
    
    # Create date field from datetime
    df_mids_tweets['tweet_date'] = df_mids_tweets.tweet_datetime.dt.date

end = datetime.now()
print('Overall: ' + str(end-start))

# Distribute Midlands circle tweets according to tweet_place/tweet_coordinates 
# (if present) or user_location
# Only done using user_location so far, tweet_place/tweet_coordinates will be a
# bit more complicated
# Current method puts tweets from users with location e.g. 'Derbyshire' into 
# Derby and Derbyshire due to 'contains'
for area in df_utlas.drop_duplicates(subset=['utla']).utla:
    print(area)
    manual_assign_areas(df_mids_tweets,df_tweets)  

# Deduplicate by tweet_id and utla
df_tweets_deduped = df_tweets.drop_duplicates(subset=['tweet_id','area'])

# UTLA frequencies
print(pd.value_counts(df_tweets.area))
print(pd.value_counts(df_tweets_deduped.area))
print(pd.value_counts(df_tweets_deduped.tweet_date))

# Output to csvs
df_tweets.to_csv('{0}df_tweets_tweepy_{1}.csv'.format(filepath,
                 str(datetime.now().date())))
df_tweets_deduped.to_csv('{0}df_tweets_deduped_tweepy_{1}.csv'.format(filepath,
                         str(datetime.now().date())))
df_mids_tweets.to_csv('{0}df_mids_tweets_tweepy_{1}.csv'.format(filepath,
                      str(datetime.now().date())))
df_mids_tweets.to_csv('{0}df_timings_{1}.csv'.format(filepath,
                      str(datetime.now().date())))
