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

# Read in df_utlas from csv (for now, will do this all within Python eventually)
df_utlas = pd.read_csv("df_utlas_90.csv")

#%% For real

# Set since and until for entire period
since = '2020-03-11'
until = '2021-12-21'

searchTerms = ['vaccines','vaccine','Johnson and Johnson','vaccinated',
               'vaccination','booster','pfizer','J&J','vaccinations',
               'unvaccinated']

# Read in df_utlas from csv (for now, will do this all within Python eventually)
#df_utlas = pd.read_csv("df_utlas.csv")
#df_utlas_ = df_utlas[df_utlas.utla.isin(['Nottingham','Derby'])]
#df_utlas_ = df_utlas[df_utlas.utla.isin(['Derbyshire'])]

# Create empty dataframes
df_tweets = pd.DataFrame()
df_error = pd.DataFrame()

start = datetime.now()
# Loop through UTLAs
for utla in df_utlas.drop_duplicates(subset=['utla']).utla:
    
    # Loop through circles
    df_utla = df_utlas[df_utlas.utla == utla]#.head(5)
    
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
                        df['term'] = term
                        df['circle'] = c
                        df_tweets = df_tweets.append(df)
                        print('success')
                        print('')
                        break
            
# Dedupe tweets
df_tweets_deduped = df_tweets.drop_duplicates(subset=['url','utla'])
pd.value_counts(df_tweets_deduped.utla)
end = datetime.now()

df_tweets.to_csv('df_tweets_90.csv')

pd.value_counts(df_tweets.utla)
pd.value_counts(df_tweets.term)
pd.value_counts(df_tweets_deduped.utla)
pd.value_counts(df_tweets_deduped.term)

# Remove tweets that don't contain the term (these must be ones where the
# user profile contains the term instead)



df_tweets_deduped_term = (df_tweets_deduped[df_tweets_deduped['content'].str
                                            .lower().str.contains('|'
                                                  .join(searchTerms))]
                                            )

df_tweets_deduped['lower'] = df_tweets_deduped['content'].str.lower()

df_tweets_deduped.to_csv('df_tweets_deduped.csv')

df_tweets_deduped.iloc[583]['lower']

df_tweets_deduped_noterm = (df_tweets_deduped[~df_tweets_deduped['content'].str
                                            .lower().str.contains('|'
                                                  .join(searchTerms))]
                                            )

pd.value_counts(df_tweets_deduped_noterm.term)

df__ = df_tweets_deduped_noterm[df_tweets_deduped_noterm['term']=='vaccines']
# Both have accents above e
df__ = df_tweets_deduped_noterm[df_tweets_deduped_noterm['term']=='J&J']

df_tweets_deduped_term = (df_tweets_deduped[~df_tweets_deduped['content'].str.
                                            contains(df_tweets_deduped['term'])
                                            ])

df__ = df_tweets_deduped_term[df_tweets_deduped_term['term']=='Johnson and Johnson']

pd.value_counts(df_tweets_deduped_term.term)
# Turns out there are no actual 'J&J' tweets and only 70 'Johnson and Johnson'
# ones. Most/all of the Johnson and Johnson ones are about Boris Johnson, not
# Johnson and Johnson. Revisit search terms.

# For now output df_tweets_deduped_term to csv to use later
df_tweets_deduped_term.to_csv('df_tweets_deduped_term.csv')
