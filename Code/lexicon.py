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

#%% LRSentiA

from SSentiA import LRSentiA
class_lr = LRSentiA.LexicalAnalyzer()

# Drop label - only in the input to make it work
#df_in = df_in.head(20)
data_LRSentiA, predictions_LRSentiA, pred_confidence_scores_LRSentiA, total_positive_score_LRSentiA, total_negative_score_LRSentiA, tweet_ids = class_lr.main(df_in)

#%% VADER

import pandas as pd
import re
#import emotlib
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# import necessary libraries
import nltk
# tokenization
nltk.download('punkt')
from nltk.tokenize import word_tokenize
# POS tagging
from nltk import pos_tag
# to map pos tags to wordnet tags
from nltk.corpus import wordnet
# stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
# Stemming
from nltk.stem import PorterStemmer
# Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
analyzer = SentimentIntensityAnalyzer()

# 1) Initial cleaning: removal of URLs, Twitter handles, punctuation, numbers,
#   single characters and multiple spaces, and conversion of emoticons to 
#   strings that convey their sentiment.
# Define a function to clean the text
def clean(text):
    # Add a space to the end of the text
    text = text + ' '
    # Remove twitter handles (any text between an @ and a space)
    text = re.sub('(?<=\@)(.*?)(?=\ )','',text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'bit.ly\S+', '', text)
    # Convert emojis to text (start by doing this before removing all non-text
    # characters, not sure how it will go)
    emojis = ''.join(c for c in text if c in emoji.UNICODE_EMOJI['en'])
    #emojis_text = emotlib.demojify(emojis)
    #emojis_text = re.sub(r':',' ',emojis_text)
    # Remove all special characters (apart from apostrophes) and numbers
    text = re.sub("[^A-Za-z'"+emojis+"]+", ' ', text)
    # Remove 
    return text

# 3)	Tokenisation: the splitting of each string of words into individual 
#   tokens.
def tokenize(text):
    text = word_tokenize(text)
    return text

# 4)	Stemming: the removal of prefixes and suffixes to reduce a word to a 
#   base word (e.g. disappointing becomes disappoint, caring becomes car).

# Stemming and lemmatisation code adapted from here: https://github.com/harika-bonthu/StemVsLemma/blob/main/stem_lemma.ipynb

def stem(text):
    stemmer = PorterStemmer()
    stem = ''
    for ele in text:
        if ele not in stopwords.words('english'):
            stem = stem + ' ' + ele
    return stem

# 5)	Lemmatisation: lemmatisation returns the dictionary form of the target 
#   word and can be preferable to stemming (e.g. better becomes good, caring 
#   becomes care) [88].

# Stop words include words that change the meaning of a sentence (e.g. didn't),
# so don't remove them.

def lemmatise(text):
    # POS tagger dictionary
    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    
    wordnet_lemmatizer = WordNetLemmatizer()
    
    lemma = ''
    pos = pos_tag(text)
    for ele, tag in pos:
        tag = pos_dict.get(tag[0])
        if not tag:
            lemma = lemma + ' ' + ele
        else:
            lemma = lemma + ' ' + wordnet_lemmatizer.lemmatize(ele, tag)
    return lemma

# Return compound polarity scores
def VADERsentiment(text):
    vs = analyzer.polarity_scores(text)
    return vs['compound']

# Calculate score confidence
def sentconf(text):
    pos = 0
    neg = 0
    if len(text.split()) > 0: #there are a very small number of tweets that just contain handles and images
        for i in range(len(text.split()) + 1):
            #print(i)
            if i == 0:
                pair = text.split()[i]
            elif i < len(text.split()):
                pair = text.split()[i-1] + ' ' + text.split()[i]
            else:
                pair = text.split()[i-1]
            #print(pair)
            #print(analyzer.polarity_scores(pair))
            pos += analyzer.polarity_scores(pair)['pos']
            neg += analyzer.polarity_scores(pair)['neg']
        if (pos+neg)>0: #if there are no positive or negative words, conf should be zero
            conf = abs(pos-neg)/(pos+neg)
        else:
            conf = 0
        return conf

# Process data
df_in['content_lemma'] = df_in['tweet_text'].apply(clean).apply(tokenize).apply(lemmatise)
df_out = df_in.copy()
df_out['sentiment'] = df_in['content_lemma'].apply(VADERsentiment)
df_out['sentconf'] = df_out['content_lemma'].apply(sentconf)

# This takes a few seconds, whereas LRSentiA takes hours.

df_out.to_csv("INDP//Data//df_VADER.csv")
