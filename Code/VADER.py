# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:14:21 2022

@author: Joe.WozniczkaWells
"""

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

class VADER():
    
    # Initial cleaning: removal of URLs, Twitter handles, punctuation, numbers,
    # single characters and multiple spaces
    # text: text to be cleaned
    def clean(self,text):
        # Add a space to the end of the text
        text = text + ' '
        # Remove twitter handles (any text between an @ and a space)
        text = re.sub('(?<=\@)(.*?)(?=\ )','',text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'bit.ly\S+', '', text)
        # Identify emojis, to prevent them from being removed
        emojis = ''.join(c for c in text if c in emoji.UNICODE_EMOJI['en'])
        # Remove all special characters (apart from apostrophes) and numbers
        text = re.sub("[^A-Za-z'"+emojis+"]+", ' ', text)
        # Remove 
        return text
    
    # Tokenisation: split each string of words into individual tokens
    # text: text to be tokenized
    def tokenize(self,text):
        text = word_tokenize(text)
        return text
    
    # Stemming: the removal of prefixes and suffixes to reduce a word to a 
    #   base word (e.g. disappointing becomes disappoint, caring becomes car)
    # text: text to be stemmed    
    def stem(self,text):
        stemmer = PorterStemmer()
        stem = ''
        for ele in text:
            if ele not in stopwords.words('english'):
                stem = stem + ' ' + ele
        return stem

    # Remove some words from stopwords, where they have an impact on the 
    # sentiment. Use the word 'good' as a test case for whether a stop word 
    # changes the sentiment.
    # stopwords_in: initial list of stopwords
    def fin_stopwords(self,stopwords_in):
        analyzer = SentimentIntensityAnalyzer()
        remove_list = []
        for word in stopwords_in:
            if (analyzer.polarity_scores(word + " good")['compound'] != 
                analyzer.polarity_scores("good")['compound']):
                remove_list.append(word)
        
        stopwords_list = [ele for ele in stopwords.words('english') if ele not in remove_list]
        return stopwords_list
    
    # Lemmatisation: returns the dictionary form of the target word 
    # (e.g. better becomes good, caring becomes care)
    # text: text to be lemmatised
    # stopwords_list: list of stopwords to be ignored
    def lemmatise(self,text,stopwords_list):
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
    # text: text to be sentiment analysed
    def VADERsentiment(self,text):
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(text)
        return vs['compound']
    
    # Calculate score confidence
    # text: text for which to return sentiment confidence score
    def sentconf(self,text):
        analyzer = SentimentIntensityAnalyzer()
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

    