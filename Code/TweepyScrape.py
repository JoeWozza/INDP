# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:10:23 2022

@author: Joe.WozniczkaWells
"""

import pandas as pd
import tweepy
from datetime import datetime

class TweetScrape():
    
    def tweepy_scrape(self,api,df,df_timings,term,lat,long,radius,area,
                      area_circle):
        # Uses tweepy's api.search_tweets function to scrape tweets and add 
        # them to an existing dataframe, df. The function outputs df and a 
        # separate dataframe, df_timings, that captures information on the 
        # volumes of tweets downloaded and time taken.
        # api: valid tweepy.api.API object
        # df_tweets: name of dataframe to contain tweets
        # df_timings: name of dataframe to contain timings
        # term: search term to be used to find tweets
        # lat: latitude of circle centroid to be used to find tweets
        # long: longitude of circle centroid to be used to find tweets
        # radius: radius of circle centroid to be used to find tweets
        # area: name of geographical area
        # area_circle: name/number of geographical area circle
        
        print(area + ': circle ' + area_circle + ' - ' + term)
        
        term_start = datetime.now()
        # Collect tweets in df_tweets
        tweets = 0
        for tweet in tweepy.Cursor(api.search_tweets, q=term, 
                                   geocode='{0},{1},{2}km'.format(lat,long,
                                            radius),
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
    
    def tweepy_scrape_terms(self,api,df_tweets,df_timings,searchTerms,lat,long,
                            radius,area,area_circle):
        # Loops through search terms defined in searchTerms and downloads
        # tweets.
        # api: valid tweepy.api.API object
        # df_tweets: name of dataframe to contain tweets
        # df_timings: name of dataframe to contain timings
        # searchTerms: list of search terms to be searched for on Twitter
        # lat: latitude of circle centroid to be used to find tweets
        # long: longitude of circle centroid to be used to find tweets
        # radius: radius of circle centroid to be used to find tweets
        # area: name of geographical area
        # area_circle: name/number of geographical area circle
        for term in searchTerms:
            df_tweets, df_timings = self.tweepy_scrape(api,df_tweets,
                                                       df_timings,term,lat,
                                                       long,radius,area,
                                                       area_circle)
        return df_tweets, df_timings
    
    def check_circle(self,api,check_term,lat,long,radius):
        # Checks whether the geographical circle contains any tweets containing
        # a user-defined search term.
        # api: valid tweepy.api.API object
        # check_term: search term used to check whether tweets exist
        # lat: latitude of circle centroid to be used to find tweets
        # long: longitude of circle centroid to be used to find tweets
        # radius: radius of circle centroid to be used to find tweets
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
    
    def circles_scrape(self,api,area,df_area,df_tweets,df_timings,searchTerms):
        # Loops through defined circles that make up an area and downloads 
        # tweets.
        # api: valid tweepy.api.API object
        # area: name of geographical area
        # df_area: dataframe containing circles to loop through
        # df_tweets: existing dataframe to append tweets to
        # df_timings: existing dataframe to append timings to
        
        # Loop through circles
        for c,(lat,long,radius) in enumerate(zip(df_area.lat, df_area.long, 
              df_area.radius)):
            
            # Are there any tweets from the circle? Test using search term 
            # 'twitter'.
            circ_tweets = self.check_circle(api,'twitter',lat,long,radius)
            
            # If there are no tweets in the circle, skip it, otherwise search 
            # for tweets.
            if circ_tweets == False:
                print('Skip circle ' + str(c+1) + ' in ' + area)
            else:
                # Loop through search terms
                df_tweets, df_timings = self.tweepy_scrape_terms(api,df_tweets,
                                                                 df_timings,
                                                                 searchTerms,
                                                                 lat,long,
                                                                 radius,area,
                                                                 str(c+1))
    
        # Create date field from datetime
        df_tweets['tweet_date'] = df_tweets.tweet_datetime.dt.date
        
        return df_tweets, df_timings
    
    def circles_scrape_areas(self,api,areas,df_areas,area_col,df_tweets,
                             df_timings,searchTerms):
        # Loops through areas and the circles they are made up of and downloads
        # tweets
        # api: valid tweepy.api.API object
        # areas: list of areas to download tweets for
        # df_areas: dataframe containing area and the lat, long and radius of 
        #   the circles that are used to approximate the area
        # area_col: name of column containing area in df_areas
        # df_tweets: existing dataframe to append tweets to
        # df_timings: existing dataframe to append timings to
        # searchTerms: list of search terms to be searched for on Twitter
        for area in areas:
            df_area = df_areas[df_areas[area_col] == area]#.head(5)
    
            df_tweets, df_timings = self.circles_scrape(api,area,df_area,
                                                        df_tweets,df_timings,
                                                        searchTerms)
            
        return df_tweets, df_timings
    
    def manual_assign_areas(self,df_in,areas):
        # Assigns tweets to an area based on whether user_location contains the
        # area name
        # df_in: dataframe containing tweets
        # areas: list of areas to assign tweets to
        for area in areas:
            print(area)
            df_temp = df_in[df_in.user_location.str.contains(area, 
                                                             na=False)].copy()
            df_temp.area = area
        
        return df_temp

