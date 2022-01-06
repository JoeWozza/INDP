# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 18:41:37 2021

@author: Joe.WozniczkaWells
"""

import pandas as pd
import re
import emotlib
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
from nltk.tokenize import word_tokenize

#df = pd.read_csv('df_tweets_example.csv')
df = pd.read_csv('df_tweets_deduped_term.csv')

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
    #emojis_text = emotlib.demojify(emojis)
    #emojis_text = re.sub(r':',' ',emojis_text)
    # Remove all special characters (apart from apostrophes) and numbers
    text = re.sub("[^A-Za-z'"+emojis+"]+", ' ', text)
    # Remove 
    return text

# Cleaning the text in the review column
#df['content_cleaned'] = df['content'].apply(clean)

# May have to look into emojis more. See https://stackoverflow.com/questions/57744725/how-to-convert-emojis-emoticons-to-their-meanings-in-python

# 2)	Spellcheck: Tweets often contain spelling mistakes or colloquialisms (e.g. yeeeeeeeeesssss) that can be identified and corrected using a spellchecker. Research has found this can be detrimental [87].

# Leave this for now

# 3)	Tokenisation: the splitting of each string of words into individual 
#   tokens.

def tokenize(text):
    text = word_tokenize(text)
    return text

#df['content_tokenized'] = df['content_cleaned'].apply(tokenize)

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

#df['content_stemmed'] = df['content_tokenized'].apply(stem)

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

#df['content_lemma'] = df['content_tokenized'].apply(lemmatise)

def lemmatise_incstop(text):
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

#df['content_lemma_incstop'] = df['content_tokenized'].apply(lemmatise_incstop)

df['content_lemma_incstop'] = df['content'].apply(clean).apply(tokenize).apply(lemmatise_incstop)

# 6)	Identify negation: identifying words that negate the meaning of other 
#   words (e.g. the piano is not heavy) is important and can prevent sentiment 
#   analysis from returning incorrect polarity scores. Prefixing the words 
#   after negation with ‘NOT_’ has been found to be preferable to replacing 
#   target words with antonyms [89]

# This is handled by the VADER algorithm.

# 7)	Stop word removal: exclusion of words that do not carry an opinion or 
#   have an impact on polarity (e.g. these, where).

# This has been done during stemming and lemmatisation.

#%% Test sentiment analysis on content and content_lemma
analyzer = SentimentIntensityAnalyzer()
def VADERsentiment(text):
    vs = analyzer.polarity_scores(text)
    return vs['compound']

#df['VaderSentiment_lemma'] = df['content_lemma'].apply(VADERsentiment)
#df['VaderSentiment_lemma_incstop'] = df['content_lemma_incstop'].apply(VADERsentiment)
#df['VaderSentiment_stemmed'] = df['content_stemmed'].apply(VADERsentiment)
#df['VaderSentiment'] = df['content'].apply(VADERsentiment)

#df.to_csv('df_sentiment.csv')
# Think the scores on the lemma_incstop versions are better.

#df_means = df.groupby(['utla']).mean()
#df_term_means = df.groupby(['term','utla']).mean()

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

df['sentiment'] = df['content_lemma_incstop'].apply(VADERsentiment)
df['sentconf'] = df['content_lemma_incstop'].apply(sentconf)

#%%

analyzer.polarity_scores('think tweet Vaccine rolling_on_the_floor_laughing rolling_on_the_floor_laughing')
analyzer.polarity_scores('🤣')
analyzer.polarity_scores('didn’t think that through before you tweeted did you? So what’s the Vaccine for?')

analyzer.polarity_scores('Booster done... smacked off my tits on Moderna now!!! 😵 #vaccine #booster')
analyzer.polarity_scores('😵 ')
analyzer.polarity_scores('dizzy_face')

analyzer.polarity_scores('😓 ')
analyzer.polarity_scores('I am happy')

analyzer.polarity_scores("won t")
analyzer.polarity_scores("win t")
analyzer.polarity_scores("wont")
analyzer.polarity_scores("won't")
analyzer.polarity_scores("accept")
analyzer.polarity_scores("won't accept")
analyzer.polarity_scores("Wow")
analyzer.polarity_scores("how")
analyzer.polarity_scores("sad")
analyzer.polarity_scores("and")
analyzer.polarity_scores("discriminatory")
analyzer.polarity_scores("is")
analyzer.polarity_scores("this")

analyzer.polarity_scores("Wow")
analyzer.polarity_scores("Wow how")
analyzer.polarity_scores("how sad")
analyzer.polarity_scores("sad and")
analyzer.polarity_scores("and discriminatory")
analyzer.polarity_scores("discriminatory is")
analyzer.polarity_scores("is this")
analyzer.polarity_scores("this")

analyzer.polarity_scores("Wow")
analyzer.polarity_scores("Wow how")
analyzer.polarity_scores("how incredible")
analyzer.polarity_scores("incredible and")
analyzer.polarity_scores("and amazing")
analyzer.polarity_scores("amazing is")
analyzer.polarity_scores("is this")
analyzer.polarity_scores("this shit")
analyzer.polarity_scores("shit")

analyzer.polarity_scores("Wow shit")

analyzer.polarity_scores("Not")
analyzer.polarity_scores("Not joking")
analyzer.polarity_scores("joking this")
analyzer.polarity_scores("this vaccine")
analyzer.polarity_scores("vaccine is")
analyzer.polarity_scores("is great")
analyzer.polarity_scores("great")

analyzer.polarity_scores("not great")

analyzer.polarity_scores("Not joking this vaccine is great")

analyzer.polarity_scores("Wow how sad")
analyzer.polarity_scores("how sad and")
analyzer.polarity_scores("sad and discriminatory")
analyzer.polarity_scores("and discriminatory is")
analyzer.polarity_scores("discriminatory is this")


sen1 = analyzer.polarity_scores("Wow how sad and discriminatory is this")
sen2 = analyzer.polarity_scores("Millions of Sputnik has been safely administered across the world")

conf = ((abs(sen1['pos']-.5+sen2['pos']-.5+sen1['neg']-.5+sen2['neg']-.5)/
         (abs(sen1['pos']-.5+sen2['pos']-.5)+(abs(sen1['neg']-.5+sen2['neg']-.5))))
    )

conf = (abs(sen1['pos']+sen2['pos']-sen1['neg']-sen2['neg'])/(sen1['pos']+sen2['pos']+sen1['neg']+sen2['neg']))
# This this (^) is the correct calculation

sent = "Wow how sad and discriminatory is this Millions of Sputnik has been safely administered across the world"
sent = "Vaccines are the shit"
sent = "Vaccines are shit"
sent = "Vaccines are dangerous"
sent = "Vaccines are not dangerous"
sent = "Vaccines are safe"

pos = 0
neg = 0
for i in range(len(sent.split()) + 1):
    print(i)
    if i == 0:
        pair = sent.split()[i]
    elif i < len(sent.split()):
        pair = sent.split()[i-1] + ' ' + sent.split()[i]
    else:
        pair = sent.split()[i-1]
    print(pair)
    print(analyzer.polarity_scores(pair))
    pos += analyzer.polarity_scores(pair)['pos']
    neg += analyzer.polarity_scores(pair)['neg']
conf = abs(pos-neg)/(pos+neg)
print(conf)

analyzer.polarity_scores(sent)

analyzer.polarity_scores("Wow how sad and discriminatory is this Millions of Sputnik has been safely administered across the world but won t accept it but it will accept their biggest trading partners China s vaccine And people still think this is not political Wish you well Nata  folded_hands ")
analyzer.polarity_scores("Wow how sad and discriminatory is this Millions of Sputnik has been safely administered across the world but wont accept it but it will accept their biggest trading partners China s vaccine And people still think this is not political Wish you well Nata  folded_hands ")
analyzer.polarity_scores("Wow how sad and discriminatory is this Millions of Sputnik has been safely administered across the world but won't accept it but it will accept their biggest trading partners China's vaccine And people still think this is not political Wish you well Nata  folded_hands ")
# Need to make sure to keep some stop words (e.g. won't) that change meaning of a sentence

stopwords.words('english')

analyzer.polarity_scores(lemmatise(['I', 'am', 'not', 'happy']))
analyzer.polarity_scores(lemmatise(['I am not happy']))
analyzer.polarity_scores(lemmatise(['I', 'am', 'sad', 'not', 'happy']))
analyzer.polarity_scores(lemmatise(['I am sad not happy']))
analyzer.polarity_scores(lemmatise(['I am happy not sad']))

analyzer.polarity_scores(lemmatise(['I am sad not happy 🤣']))
analyzer.polarity_scores(['happy 🤣'])
analyzer.polarity_scores(['🤣'])
analyzer.polarity_scores(['happy'])
analyzer.polarity_scores(['sad 🤣'])
analyzer.polarity_scores(['🤣 sad'])
analyzer.polarity_scores(lemmatise(['I am sad 🤣 not happy 🤣 🤣  ']))
analyzer.polarity_scores(lemmatise(['I am happy not sad 🤣']))
analyzer.polarity_scores(lemmatise(['happy', 'am', 'sad', 'I', 'not']))

analyzer.polarity_scores(lemmatise(['st vaccine jab time hopefully first step kind normality']))
analyzer.polarity_scores(lemmatise(['st vaccine jab time and hopefully a first step to some kind of normality']))

analyzer.polarity_scores(lemmatise(['kind of normality']))
analyzer.polarity_scores(lemmatise(['kind normality']))

# Not sure about removing stop words. 'not' is a stop word, so 'I am not happy'
# and 'I am happy' are both lemmatised to 'happy', which is bad.
pd.value_counts(df[df['utla']=='Derby'].username)

df_booster = df[df['term'] == 'booster']


