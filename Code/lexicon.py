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
# some have 86444 records, predictions_LRSentiA, total_negative_score_LRSentiA,
# total_positive_score_LRSentiA and tweet_ids only have 86443.

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
# so don't remove all of them.
# Remove some words from stopwords, where they have an impact on the sentiment.
# Use the word 'good' as a test case for whether a stop word changes the
# sentiment.
remove_list = []
for word in stopwords.words('english'):
    if analyzer.polarity_scores(word + " good")['compound'] != analyzer.polarity_scores("good")['compound']:
        remove_list.append(word)

stopwords_list = [ele for ele in stopwords.words('english') if ele not in remove_list]

def lemmatise(text):
    # POS tagger dictionary
    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    
    wordnet_lemmatizer = WordNetLemmatizer()
    
    lemma = ''
    pos = pos_tag(text)
    for ele, tag in pos:
        if ele not in stopwords_list:
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
            #print(pos)
            #print(neg)
        if (pos+neg)>0: #if there are no positive or negative words, conf should be zero
            conf = abs(pos-neg)/(pos+neg)
        else:
            conf = 0
        return conf

# Process data
df_VADER = df_tweets_deduped_eng.copy()
df_VADER['tweet_id'] = df_VADER['tweet_id'].astype('Int64').apply(str)
df_VADER['content_lemma'] = df_tweets_deduped_eng['tweet_text'].apply(clean).apply(tokenize).apply(lemmatise)
df_VADER['sentiment'] = df_VADER['content_lemma'].apply(VADERsentiment)
df_VADER['sentconf'] = df_VADER['content_lemma'].apply(sentconf)

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