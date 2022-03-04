# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 09:35:27 2021

@author: Joe.WozniczkaWells
"""

from os import chdir

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

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

def sntwitter_scrape(df_tweets,df_error,area,term,lat,long,radius,since,until,
                     area_circle):
    # Try 40 times
    for x in range(0, 40):
        try:
            df = pd.DataFrame(sntwitter.TwitterSearchScraper(
                    '{0} geocode:"{1},{2},{3}km" since:{4} until:{5}'.format(
                            term,lat,long,radius,since,until)).get_items())
            str_error = None
        except:
            str_error = Exception
            pass
        if str_error:
            # Wait for 2 seconds before trying to fetch the data again
            sleep(2)
            if x == 39:
                dict_error = {'term': term, 'lat': lat, 'long': long, 
                              'radius': radius, 'since': since, 'until': until}
                df_error = df_error.append(dict_error, ignore_index=True)
            print('error')
        else:
            df['area'] = area
            df['search_term'] = term
            df['area_circle'] = area_circle
            df_tweets = df_tweets.append(df)
            print('success')
            print('')
            break
    
    return df_tweets, df_error

def sntwitter_scrape_terms(df_tweets,df_error,df_area,searchTerms,area,lat,
                           long,radius,since,until,area_circle):
    for term in searchTerms:
        print(area,': circle',area_circle+1,'of',len(df_area))
        print(term)
        print(since,'to',until)
        df_tweets, df_error = sntwitter_scrape(df_tweets,df_error,area,term,
                                               lat,long,radius,since,until,
                                               area_circle)
    
    return df_tweets, df_error

def sntwitter_scrape_circles(df_tweets,df_error,area,df_area,since,until,
                             searchTerms):
    for c,(lat,long,radius) in enumerate(zip(df_area.lat, df_area.long, 
          df_area.radius)):
        df_tweets, df_error = sntwitter_scrape_terms(df_tweets,df_error,
                                                     df_area,searchTerms,area,
                                                     lat,long,radius,since,
                                                     until,c)
    
    return df_tweets, df_error

def sntwitter_scrape_areas(df_tweets,df_error,df_areas,area_col,since,until,
                           searchTerms):
    for area in df_areas.drop_duplicates(subset=[area_col])[area_col]:
        print(area)
        # Loop through circles
        df_area = df_areas[df_areas[area_col] == area]
        
        df_tweets, df_error = sntwitter_scrape_circles(df_tweets,df_error,area,
                                                       df_area,since,until,
                                                       searchTerms)
        df_tweets = df_tweets.rename(columns={'content':'tweet_text',
                                              'date':'tweet_datetime',
                                              'id':'tweet_id',
                                              'username':'user_name'})
        # Remove tweets that do not contain any of the search terms (sometimes
        # sntwitter downloads tweets where the search term is in the user's 
        # bio)
        df_tweets = (df_tweets[df_tweets['tweet_text'].str.lower().str.
                               contains('|'.join(searchTerms))])
    
    return df_tweets, df_error

# Create empty dataframes
df_tweets = pd.DataFrame()
df_error = pd.DataFrame()
# Set since and until for entire period
since = '2020-03-11'
until = datetime.now().strftime("%Y-%m-%d")

df_utlas = df_utlas[df_utlas['utla']=='Derby']
df_tweets, df_error = sntwitter_scrape_areas(df_tweets,df_error,df_utlas,
                                             'utla',since,until,searchTerms)