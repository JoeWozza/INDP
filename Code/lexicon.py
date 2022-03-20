# -*- coding: utf-8 -*-
"""
This code uses the VADER class to get sentiment scores (-1 to 1) and sentiment
confidence scores (0 to 1) for Tweets captured in several dated csv files.

@author: Joe Wozniczka-Wells
"""

from os import chdir, getcwd,listdir

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

# Load packages
import pandas as pd
import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
analyzer = SentimentIntensityAnalyzer()
from INDP.Code import VADER
class_v = VADER.VADER()
import os

# Create folders in which to save VADER output
data_folder = 'INDP/Data'
vader_folder = 'INDP/Data/VADER'
tweet_folder = 'INDP/Data/Tweets'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)
if not os.path.exists(vader_folder):
    os.makedirs(vader_folder)

#%% Read in data

df_tweets_eng = pd.read_csv("{0}/df_tweets_eng.csv".format(tweet_folder))
    
# Deduplicate by tweet_id
df_VADER = (df_tweets_eng
            .drop_duplicates(subset=['tweet_id']).reset_index()
            .drop(columns=['index','Unnamed: 0']))

#%% VADER

# Define stopwords list
stopwords_fin = class_v.fin_stopwords(stopwords.words('english'))
# Sort out tweet_id field
df_VADER['tweet_id'] = df_VADER['tweet_id'].astype('Int64').apply(str)
# Clean, tokenize and lemmatise tweet content
df_VADER['content_lemma'] = (df_VADER['tweet_text'].apply(class_v.clean)
    .apply(class_v.tokenize)
    .apply(class_v.lemmatise, stopwords_list=stopwords_fin))
# Calculate sentiment
df_VADER['sentiment'] = df_VADER['content_lemma'].apply(class_v.VADERsentiment)
# Calculate sentiment confidence
df_VADER['sentconf'] = df_VADER['content_lemma'].apply(class_v.sentconf)
# Categorise sentiment score
df_VADER['sentiment_cat'] = df_VADER.apply(lambda row: 
    class_v.cat_sentiment(row['sentiment']), axis=1)
# Categorise sentiment confidence score
mean_cs,std_cs = class_v.cat_sentconf_stats(df_VADER['sentconf'])
df_VADER['sentconf_cat'] = df_VADER.apply(lambda row:
    class_v.cat_sentconf(row['sentconf'],mean_cs,std_cs), axis=1)

# Output results to csv
df_VADER.to_csv("{0}/df_VADER.csv".format(vader_folder))
