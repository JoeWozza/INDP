# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 09:35:27 2021

@author: Joe.WozniczkaWells
"""

import pandas as pd
import snscrape.modules.twitter as sntwitter
import itertools

from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from time import sleep

def datespan(startDate, endDate, delta):
    currentDate = startDate
    while currentDate < endDate:
        yield currentDate
        currentDate += delta

# Use vaccination-related search terms from Bonnevie et al. plus some from
# Sattar & Arifuzzaman

searchTerms = [
        # Bonnevie et al.
        'antivaxmovement','antivaxxmovement','fuckvaccines','getvaccinated',
               'GlobalVaccineSummit','learntherisk','NoForcedVaccination',
               'StopForcedVaccination','StopMandatoryVaccination','vaccinateus',
               'vaccinechoice','vaccinedamage','vaccinedeath','vaccinefailure',
               'vaccineinjury','vaccineprotection','VaccinesCauseAIDs',
               'VaccinesCauseAutism','VaccinesCauseRegressiveAutism',
               'vaccinescausesids','vaccinesdangers','vaccineskill',
               'vaccineskillandmaim','Vaccinesuncovered','vaccineswork',
               'vaccinetruth','VaXXedII','anti-flu-vaccine','anti-vacc',
               'anti-vaccination','anti-vaccine','anti-vaccines','anti-vax',
               'anti-vaxer','anti-vaxers','antivaccination','antivaccine',
               'antivaccines','antivax','antivaxer','antivaxers','antivaxx','antivaxxer','antivaxxers','flu shot',
               'immunization','immunizations','pre-vax','pro-immunization',
               'pro-immunizations','pro-vaccination','pro-vaccine',
               'pro-vaccines','pro-vax','pro-vaxer','pro-vaxers','pro-vaxx',
               'pro-vaxxer','pro-vaxxers','provaccine','provaccines',
               'unvaccinated','unvax','unvaxed','unvaxxed','vacc damage','vacc schedule',
               "vacc'd","vacc'n",'vaccinate','vaccinater','vaccinates',
               'vaccinateing','vaccination','vaccinations','vaccine','vaccines',
               'vaccine-hesitant','vacine','vacines','vaxer','VaXism','vaxx',
               'vaxxed','vaxxer',
               # Sattar & Arifuzzaman
               'pfizer','Pfizer-BioNTech','BioNTechpfizer','Moderna',
               'moderna_tx','Moderna-NIAID','NIAID','NIAID-Moderna',
               'Johnson & Johnson','Johnson and Johnson','Janssen',
               'Janseen Pharmaceutical','J&J','OXFORDVACCINE',
               'Oxford-AstraZeneca','OxfordAstraZeneca','AstraZeneca',
               'Vaxzevria','Covishield','Sputnik V','sputnikv','sputnikvaccine',
               'covaxin','BharatBiotech','coronavac','sinovac',
               # Me
               'booster','first dose','second dose','third dose']

searchTerms = ['vaccination','vaccine']

# Read in df_utlas from csv (for now, will do this all within Python eventually)
df_utlas = pd.read_csv("df_utlas.csv")
df_utlas_ = df_utlas[df_utlas.utla.isin(['Nottingham','Derby'])]
#df_utlas_ = df_utlas[df_utlas.utla.isin(['Derby'])]

# Set since and until for entire period
since = '2020-03-11'
until = '2021-12-25'

# Do some investigation to see whether doing it for a longer period misses some tweets

# Create empty dataframes
df_tweets = pd.DataFrame()
df_error = pd.DataFrame()

# Loop through UTLAs
for utla in df_utlas_.drop_duplicates(subset=['utla']).utla:
    
    # Loop through circles
    df_utla = df_utlas[df_utlas.utla == utla]#.head(3)
    
    for c,(lat,long,radius) in enumerate(zip(df_utla.lat, df_utla.long, 
          df_utla.radius)):
        # Loop through search terms for tweets since 11/3/20, the day WHO 
        # declared the pandemic.
        for term in searchTerms:
            
            ## Loop through months to avoid 'Unable to find guest token'
            ## error (since is inclusive, until is exclusive)
            #for since in datespan(date(2020, 3, 11), date.today(), 
            #                      delta=relativedelta(months=+6)):
            #    until = since + relativedelta(months=+6)
                print(utla,': circle',c+1,'of',len(df_utla))
                print(term)
                print(since,'to',until)
                for x in range(0, 40):  # try 40 times
                    try:
                        df = pd.DataFrame(sntwitter.TwitterSearchScraper(
                                '{0} geocode:"{1},{2},{3}km" since:{4} until:{5}'.format(term,lat,long,radius,since,until)).get_items())
                        str_error = None
                    except:
                        str_error = Exception
                        pass
                    if str_error:
                        sleep(2)  # wait for 2 seconds before trying to fetch the data again
                        if x == 39:
                            dict_error = {'term': term, 'lat': lat, 'long': long, 
                                          'radius': radius, 'since': since, 'until': until}
                            df_error = df_error.append(dict_error)
                        print('error')
                    else:
                        df['utla'] = utla
                        df_tweets = df_tweets.append(df)
                        print('success')
                        print('')
                        break
            
# Dedupe tweets
df_tweets_deduped = df_tweets.drop_duplicates(subset=['url','utla'])
pd.value_counts(df_tweets_deduped.utla)
# The above should work but sometimes gets the 'Unable to find guest token' error
# Try tweepy instead
    
#%%
import tweepy
import numpy as np

# Keys from here: https://developer.twitter.com/en/portal/projects/1475494865567899649/apps/new
consumer_key = 'OPqZPaFbJHx8sXf8y7C5umylY'
consumer_secret = 'qE6FdosS3aiSHNDcamEhxcGCxl40k4oKoPNoVTrrW8IMWkYSkB'
# See here for keeping them secure: https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

api = tweepy.API(auth)


df_tweets = pd.DataFrame(columns = ['utla','tweet_id','tweet_text',
                                    'tweet_date','tweet_place','tweet_coords',
                                    'user_name','user_location']
                )
for tweet in tweepy.Cursor(api.search_tweets, q='vaccine', 
                           geocode='{0},{1},{2}km'.format(lat,long,radius),
                           since="2021-03-11",until='2021-12-25',
                           tweet_mode='extended').items(99999999999):
    #print(tweet)
    print(tweet.created_at)
    #print(tweet.id)
    #print(tweet.full_text)
    #print(tweet.place)
    #print(tweet.coordinates)
    #print(tweet.user.name)
    #print(tweet.user.location)
    #print('')
    dict_tweet = {'utla': utla, 'tweet_id': tweet.id, 
                  'tweet_text': tweet.full_text, 
                  'tweet_date': tweet.created_at, 'tweet_place': tweet.place, 
                  'tweet_coords': tweet.coordinates, 
                  'user_name': tweet.user.name, 
                  'user_location': tweet.user.location}
    df_tweets = df_tweets.append(dict_tweet, ignore_index=True)

# Twitter API only returns data for the previous 7 days


