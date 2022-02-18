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

# Create input
filestring_eng = 'df_tweets_eng_tweepy_'

files = listdir(filepath + "\\INDP\\Data")

tweets_files_eng = [s for s in files if filestring_eng in s]

df_tweets_eng = pd.DataFrame()

for f in tweets_files_eng:
    print(f)
    df_tweets_eng = df_tweets_eng.append(pd.read_csv("INDP\\Data\\" + f))

df_tweets_deduped_eng = df_tweets_eng.drop_duplicates(subset=['tweet_id'])
df_in = df_tweets_deduped_eng[['tweet_text','tweet_id']]
df_in['tweet_id'] = df_in['tweet_id'].astype('Int64').apply(str)

#%% VADER

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
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
analyzer = SentimentIntensityAnalyzer()

stopwords_fin = class_v.fin_stopwords(stopwords.words('english'))

# Process data
df_VADER = df_tweets_deduped_eng.head(10).copy()
df_VADER['tweet_id'] = df_VADER['tweet_id'].astype('Int64').apply(str)
df_VADER['content_lemma'] = (df_VADER['tweet_text'].apply(class_v.clean)
    .apply(class_v.tokenize)
    .apply(class_v.lemmatise, stopwords_list=stopwords_fin))
df_VADER['sentiment'] = df_VADER['content_lemma'].apply(class_v.VADERsentiment)
df_VADER['sentconf'] = df_VADER['content_lemma'].apply(class_v.sentconf)

# This takes a few minutes, whereas LRSentiA takes hours.

df_VADER.to_csv("INDP//Data//df_VADER.csv")

# This example is a problem. Compound output is positive, for some reason, but
# all pairs analysed by sentconf are either negative or neutral, so confidence
# score = 1. Removing stop words helps, because it produces positive and
# negative pairs, but this seems like a fluke.
text = "This CAN NOT be ignore The government must acknowledge this issue and either prove it invalid or act to halt vaccine for male in this age group"
analyzer.polarity_scores(text)

sentconf(text)

text_ = word_tokenize(text)
text2 = ''
for ele in text_:
    if ele not in stopwords_list:
        text2 = text2 + ' ' + ele

analyzer.polarity_scores(text2)
sentconf(text2)

stopwords_list = stopwords.words('english')
print(len(stopwords_list))
stopwords_list.remove("wasn't")
print(len(stopwords_list))

# Remove some words from stopwords, where they have an impact on the sentiment
remove_list = []
for word in stopwords.words('english'):
    if analyzer.polarity_scores(word + " good")['compound'] != analyzer.polarity_scores("good")['compound']:
        remove_list.append(word)

stopwords_list = [ele for ele in stopwords.words('english') if ele not in remove_list]
print(len(stopwords_list))
# PUT THIS (^) IN EARLIER ON TO REMOVE STOP WORDS DURING LEMMATISATION