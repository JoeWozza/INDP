# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 18:41:37 2021

@author: Joe.WozniczkaWells
"""

import pandas as pd
import re
import emotlib
import emoji

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
from nltk.tokenize import word_tokenize

df = pd.read_csv('df_tweets_example.csv')
df = pd.read_csv('df_tweets_deduped_term.csv')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df['content']]
df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df['content']]
df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df['content']]
df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df['content']]

df.to_csv('df_sentiment.csv')

# 1) Initial cleaning: removal of URLs, Twitter handles, punctuation, numbers,
#   single characters and multiple spaces, and conversion of emoticons to 
#   strings that convey their sentiment.

# sntwitter includes profile names (not handles but the bit people can change).
# Need to go through all tweets and remove those where 'content' does not
# include the term.

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
    emojis_text = emotlib.demojify(emojis)
    emojis_text = re.sub(r':',' ',emojis_text)
    # Remove all special characters and numbers
    text = re.sub('[^A-Za-z]+', ' ', text) + emojis_text
    # Remove 
    return text

# Cleaning the text in the review column
df['content_cleaned'] = df['content'].apply(clean)

# May have to look into emojis more. See https://stackoverflow.com/questions/57744725/how-to-convert-emojis-emoticons-to-their-meanings-in-python

# 2)	Spellcheck: Tweets often contain spelling mistakes or colloquialisms (e.g. yeeeeeeeeesssss) that can be identified and corrected using a spellchecker. Research has found this can be detrimental [87].

# Leave this for now

# 3)	Tokenisation: the splitting of each string of words into individual 
#   tokens.

def tokenize(text):
    text = word_tokenize(text)
    return text

df['content_tokenized'] = df['content_cleaned'].apply(tokenize)

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

df['content_stemmed'] = df['content_tokenized'].apply(stem)

# 5)	Lemmatisation: lemmatisation returns the dictionary form of the target 
#   word and can be preferable to stemming (e.g. better becomes good, caring 
#   becomes care) [88].

def lemmatise(text):
    # POS tagger dictionary
    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    
    wordnet_lemmatizer = WordNetLemmatizer()
    
    lemma = ''
    pos = pos_tag(text)
    for ele, tag in pos:
        tag = pos_dict.get(tag[0])
        if ele.lower() not in stopwords.words('english'):
            if not tag:
                lemma = lemma + ' ' + ele
            else:
                lemma = lemma + ' ' + wordnet_lemmatizer.lemmatize(ele, tag)
    return lemma

df['content_lemma'] = df['content_tokenized'].apply(lemmatise)

# 6)	Identify negation: identifying words that negate the meaning of other 
#   words (e.g. the piano is not heavy) is important and can prevent sentiment 
#   analysis from returning incorrect polarity scores. Prefixing the words 
#   after negation with ‘NOT_’ has been found to be preferable to replacing 
#   target words with antonyms [89]

# 7)	Stop word removal: exclusion of words that do not carry an opinion or 
#   have an impact on polarity (e.g. these, where).

# This has been done during stemming and lemmatisation.

#%% Test sentiment analysis on content and content_lemma
analyzer = SentimentIntensityAnalyzer()
def VADERsentiment(text):
    vs = analyzer.polarity_scores(text)
    return vs['compound']

df['VaderSentiment_lemma'] = df['content_lemma'].apply(VADERsentiment)
df['VaderSentiment_stemmed'] = df['content_stemmed'].apply(VADERsentiment)
df['VaderSentiment'] = df['content'].apply(VADERsentiment)

df.to_csv('df_sentiment.csv')



analyzer.polarity_scores('think tweet Vaccine rolling_on_the_floor_laughing rolling_on_the_floor_laughing')
analyzer.polarity_scores('🤣')
analyzer.polarity_scores('didn’t think that through before you tweeted did you? So what’s the Vaccine for?')

analyzer.polarity_scores('Booster done... smacked off my tits on Moderna now!!! 😵 #vaccine #booster')
analyzer.polarity_scores('😵 ')
analyzer.polarity_scores('dizzy_face')

analyzer.polarity_scores('😓 ')


pd.value_counts(df[df['utla']=='Derby'].username)

df_booster = df[df['term'] == 'booster']


