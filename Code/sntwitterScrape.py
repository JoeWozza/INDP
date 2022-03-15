# -*- coding: utf-8 -*-
"""
The TweetScrape class contains functions that use the snscrape package to
download Tweets from Twitter.

@author: Joe Wozniczka-Wells
"""

import pandas as pd
import snscrape.modules.twitter as sntwitter
from time import sleep

class TweetScrape():
    
    def sntwitter_scrape(self,df_tweets,df_error,area,term,lat,long,radius,
                         since,until,area_circle):
        # Uses sntwitter's TwitterSearchScraper function to scrape tweets and 
        # add them to an existing dataframe, df. The function outputs df and a 
        # separate dataframe, df_error, that captures information on any areas
        # within which Tweets were unable to be downloaded.
        # df_tweets: name of dataframe to contain tweets
        # df_error: name of dataframe to contain unsuccessful queries
        # area: name of geographical area
        # term: search term to be used to find tweets
        # lat: latitude of circle centroid to be used to find tweets
        # long: longitude of circle centroid to be used to find tweets
        # radius: radius of circle centroid to be used to find tweets
        # since: start date of tweet search (inclusive)
        # until: end date of tweet search (exclusive)
        # area_circle: name/number of geographical area circle
        
        # Try 40 times
        for x in range(0, 40):
            try:
                df = (pd.DataFrame(sntwitter.TwitterSearchScraper(
                        '{0} geocode:"{1},{2},{3}km" since:{4} until:{5}'
                        .format(term,lat,long,radius,since,until)).get_items())
                    )
                str_error = None
            except:
                str_error = Exception
                pass
            if str_error:
                # Wait for 2 seconds before trying to fetch the data again
                sleep(2)
                if x == 39:
                    dict_error = {'term': term, 'lat': lat, 'long': long, 
                                  'radius': radius, 'since': since, 
                                  'until': until}
                    df_error = df_error.append(dict_error, ignore_index=True)
                print('error')
            else:
                # If it fails 40 times, record details and move onto next 
                # circle
                df['area'] = area
                df['search_term'] = term
                df['area_circle'] = area_circle
                df_tweets = df_tweets.append(df)
                print('success')
                print('')
                break
        
        return df_tweets, df_error
    
    def sntwitter_scrape_terms(self,df_tweets,df_error,df_area,searchTerms,
                               area,lat,long,radius,since,until,area_circle):
        # Loops through search terms defined in searchTerms and downloads
        # tweets.
        # df_tweets: name of pandas dataframe to contain tweets
        # df_area: pandas dataframe containing details of circles that make up
        #   the target area
        # searchTerms: list of search terms to be searched for on Twitter
        # area: name of geographical area
        # lat: latitude of circle centroid to be used to find tweets
        # long: longitude of circle centroid to be used to find tweets
        # radius: radius of circle centroid to be used to find tweets
        # since: start date of tweet search (inclusive)
        # until: end date of tweet search (exclusive)
        # area_circle: name/number of geographical area circle
        for term in searchTerms:
            print(area,': circle',area_circle+1,'of',len(df_area))
            print(term)
            print(since,'to',until)
            df_tweets, df_error = self.sntwitter_scrape(df_tweets,df_error,
                                                        area,term,lat,long,
                                                        radius,since,until,
                                                        area_circle)
        
        return df_tweets, df_error
    
    def sntwitter_scrape_circles(self,df_tweets,df_error,area,df_area,since,
                                 until,searchTerms):
        # Loops through defined circles that make up an area and downloads 
        # tweets.
        # df_tweets: existing dataframe to append tweets to
        # df_error: name of pandas dataframe to contain unsuccessful queries
        # area: name of geographical area
        # df_area: pandas dataframe containing details of circles that make up
        #   the target area
        # since: start date of tweet search (inclusive)
        # until: end date of tweet search (exclusive)
        # searchTerms: list of search terms to be searched for on Twitter
        for c,(lat,long,radius) in enumerate(zip(df_area.lat, df_area.long, 
              df_area.radius)):
            df_tweets, df_error = self.sntwitter_scrape_terms(df_tweets,
                                                              df_error,df_area,
                                                              searchTerms,area,
                                                              lat,long,radius,
                                                              since,until,c)
        
        return df_tweets, df_error
    
    def sntwitter_scrape_areas(self,df_tweets,df_error,df_areas,area_col,since,
                               until,searchTerms):
        # Loops through areas and the circles they are made up of and downloads
        # tweets
        # df_tweets: existing dataframe to append tweets to
        # df_error: name of pandas dataframe to contain unsuccessful queries
        # df_areas: dataframe containing area and the lat, long and radius of 
        #   the circles that are used to approximate the areas
        # area_col: name of column containing area in df_areas
        # since: start date of tweet search (inclusive)
        # until: end date of tweet search (exclusive)
        # searchTerms: list of search terms to be searched for on Twitter
        for area in df_areas.drop_duplicates(subset=[area_col])[area_col]:
            print(area)
            # Loop through circles
            df_area = df_areas[df_areas[area_col] == area]
            
            df_tweets, df_error = self.sntwitter_scrape_circles(df_tweets,
                                                                df_error,area,
                                                                df_area,since,
                                                                until,
                                                                searchTerms)
        df_tweets = df_tweets.rename(columns={'content':'tweet_text',
                                              'date':'tweet_datetime',
                                              'id':'tweet_id',
                                              'username':'user_name'})
        
        return df_tweets, df_error