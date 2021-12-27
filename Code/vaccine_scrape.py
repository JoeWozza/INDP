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

# Loop through UTLAs
for utla in df_utlas_.drop_duplicates(subset=['utla']).utla:
    print(utla)
    df_tweets = pd.DataFrame()
    
    # Loop through circles
    df_utla = df_utlas[df_utlas.utla == utla]
    
    for lat,long,radius in zip(df_utla.lat, df_utla.long, df_utla.radius):
        print(lat,long,radius)
        # Loop through search terms for tweets since 11/3/20, the day WHO 
        # declared the pandemic.
        for term in searchTerms:
            print(term)
            ## Loop through months to avoid 'Unable to find guest token'
            ## error (since is inclusive, until is exclusive)
            for since in datespan(date(2020, 3, 11), date.today(), 
                                  delta=relativedelta(weeks=+1)):
                until = since + relativedelta(weeks=+1)
                
                print(since,'to',until)
                df = pd.DataFrame(sntwitter.TwitterSearchScraper(
                        '{0} geocode:"{1}, {2}, {3}km" since:{4} until:{5}'.format(term,lat,long,radius,since,until)).get_items())
                df.utla = utla
                df_tweets = df_tweets.append(df)
            
    # Dedupe tweets
    
# The above should work but sometimes gets the 'Unable to find guest token' error
# Try tweepy instead
    
#%%
import tweepy

# Keys from here: https://developer.twitter.com/en/portal/projects/1475494865567899649/apps/new
consumer_key = 'OPqZPaFbJHx8sXf8y7C5umylY'
consumer_secret = 'qE6FdosS3aiSHNDcamEhxcGCxl40k4oKoPNoVTrrW8IMWkYSkB'
# See here for keeping them secure: https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

api = tweepy.API(auth)
for tweet in tweepy.Cursor(api.search_tweets, q='vaccine', 
                           geocode='{0},{1},{2}km'.format(lat,long,radius),
                           since='2020-03-11',until='2021-12-25',
                           tweet_mode='extended').items(1):
    #print(tweet)
    print(tweet.id)
    print(tweet.full_text)
    print(tweet.place)
    print(tweet.coordinates)
    print(tweet.user.name)
    print(tweet.user.location)
    print('')



try:
    redirect_url = auth.get_authorization_url()
except tweepy.TweepError:
    print('Error! Failed to get request token.')

token = session.get('request_token')
session.delete('request_token')
auth.request_token = { 'oauth_token' : token,
                         'oauth_token_secret' : verifier }


auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)





