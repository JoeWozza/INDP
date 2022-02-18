# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 17:02:22 2022

@author: Joe.WozniczkaWells
"""

from os import chdir, getcwd,listdir

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

import pandas as pd
from INDP.Code import VADER
class_v = VADER.VADER()

#%% Read in data

# Define common string at start of file names
filestring_eng = 'df_tweets_eng_tweepy_'
# Get list of all files in path
files = listdir(filepath + "\\INDP\\Data")
# Get list of all files containing common string
tweets_files_eng = [s for s in files if filestring_eng in s]
# Combine data into single dataframe
df_tweets_eng = pd.DataFrame()
for f in tweets_files_eng:
    print(f)
    df_tweets_eng = df_tweets_eng.append(pd.read_csv("INDP\\Data\\" + f))
# Deduplicate by tweet_id
df_VADER = df_tweets_eng.drop_duplicates(subset=['tweet_id']).reset_index().drop(columns=['index','Unnamed: 0'])

#%% VADER

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

# Output results to csv
df_VADER.to_csv("INDP//Data//df_VADER.csv")
