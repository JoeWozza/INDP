# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:57:18 2022

@author: Joe.WozniczkaWells
"""

import pandas as pd
import snscrape.modules.twitter as sntwitter

from datetime import datetime
from time import sleep

class TweetScrape():
    
    def sntwitter_scrape(self,df_tweets,df_error,area,term,lat,long,radius,
                         since,until,area_circle):
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
                print(df.shape)
                df['area'] = area
                df['search_term'] = term
                df['area_circle'] = area_circle
                print(df.shape)
                print(df_tweets.shape)
                df_tweets = df_tweets.append(df)
                print('success')
                print('')
                break
        
        return df_tweets, df_error
    
    def sntwitter_scrape_terms(self,df_tweets,df_error,df_area,searchTerms,
                               area,lat,long,radius,since,until,area_circle):
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