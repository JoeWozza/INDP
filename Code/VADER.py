# -*- coding: utf-8 -*-
"""
The VADER class contains functions that use the VADER package to assign
sentiment scores and sentiment confidence scores to passages of text.

@author: Joe Wozniczka-Wells
"""

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

class VADER():
    
    def clean(self,text):
        # Initial cleaning: removal of URLs, Twitter handles, punctuation, numbers,
        # single characters and multiple spaces
        # text: text to be cleaned
        
        # Add a space to the end of the text
        text = text + ' '
        # Remove twitter handles (any text between an @ and a space)
        text = re.sub('(?<=\@)(.*?)(?=\ )','',text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'https\S+', '', text)
        text = re.sub(r'bit.ly\S+', '', text)
        # Identify emojis, to prevent them from being removed
        emojis = ''.join(c for c in text if c in emoji.UNICODE_EMOJI['en'])
        # Remove all special characters (apart from apostrophes) and numbers
        text = re.sub("[^A-Za-z'"+emojis+"]+", ' ', text)
        # Remove 
        return text
    
    def tokenize(self,text):
        # Tokenisation: split each string of words into individual tokens
        # text: text to be tokenized
        text = word_tokenize(text)
        return text

    def fin_stopwords(self,stopwords_in):
        # Removes some words from stopwords, where they have an impact on the 
        # sentiment. Uses the word 'good' as a test case for whether a stop 
        # word changes the sentiment.
        # stopwords_in: initial list of stopwords
        
        analyzer = SentimentIntensityAnalyzer()
        remove_list = []
        for word in stopwords_in:
            if (analyzer.polarity_scores(word + " good")['compound'] != 
                analyzer.polarity_scores("good")['compound']):
                remove_list.append(word)
        
        stopwords_list = [ele for ele in stopwords.words('english') if ele not in remove_list]
        return stopwords_list
    
    def lemmatise(self,text,stopwords_list):
        # Lemmatisation: returns the dictionary form of the target word (e.g. 
        # better becomes good, caring becomes care)
        # text: text to be lemmatised
        # stopwords_list: list of stopwords to be ignored
        
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

    def VADERsentiment(self,text):
        # Return compound polarity scores
        # text: text to be sentiment analysed
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(text)
        return vs['compound']
    
    def sentconf(self,text):
        # Calculate score confidence
        # text: text for which to return sentiment confidence score
        analyzer = SentimentIntensityAnalyzer()
        pos = 0
        neg = 0
        # Exclude tweets that just contain handles and images
        if len(text.split()) > 0:
            for i in range(len(text.split()) + 1):
                if i == 0:
                    pair = text.split()[i]
                elif i < len(text.split()):
                    pair = text.split()[i-1] + ' ' + text.split()[i]
                else:
                    pair = text.split()[i-1]
                pos += analyzer.polarity_scores(pair)['pos']
                neg += analyzer.polarity_scores(pair)['neg']
            # If there are no positive or negative words, conf should be zero
            if (pos+neg)>0:
                conf = abs(pos-neg)/(pos+neg)
            else:
                conf = 0
            return conf
    
    def cat_sentiment(self,var):
        # Categorise sentiment
        # var: series containing sentiment score variable (-1 <= var <= 1)
        
        # Positive
        if var >= 0.05:
            return 1
        # Negative
        elif var <= - 0.05:
            return -1
        # Neutral
        else:
            return 0
    
    def cat_sentconf_stats(self,var):
        # Calculate thresholds for categorising sentiment confidence
        # var: series containing sentiment confidence variable
        
        mean = var.mean()
        std = var.std()
        return mean, std
    
    def cat_sentconf(self,var,mean_cs,std_cs):
        # Categorise sentiment confidence (in accordance with Sazzed & Jayarathna, 
        # 2021)
        # var: series containing sentiment confidence variable
        # mean_cs: mean confidence score
        # std_cs: standard deviation of confidence score
        
        if var >= mean_cs + 0.5*std_cs:
            return 'VeryHigh'
        elif var >= mean_cs + 0.5*std_cs - 0.5*std_cs:
            return 'High'
        elif var >= mean_cs + 0.5*std_cs - std_cs:
            return 'Low'
        elif var > 0:
            return 'VeryLow'
        else:
            return 'Zero'

    