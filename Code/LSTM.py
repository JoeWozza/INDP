# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:33:59 2022

@author: Joe.WozniczkaWells
"""

import pandas as pd
##
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, median_absolute_error
#import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tkr
from datetime import datetime
import pickle

class LSTM():
    
    # Tokenize and lemmatize texts
    # text_list: list of texts to tokenize and lemmatize
    # Code adapted from https://towardsdatascience.com/sentiment-analysis-comparing-3-common-approaches-naive-bayes-lstm-and-vader-ab561f834f89
    def data_cleaning(self,text_list):
        lemmatizer=WordNetLemmatizer()
        tokenizer=TweetTokenizer()
        reconstructed_list=[]
        for each_text in text_list:
            lemmatized_tokens=[]
            tokens=tokenizer.tokenize(each_text.lower())
            pos_tags=pos_tag(tokens)
            for each_token, tag in pos_tags: 
                if tag.startswith('NN'): 
                    pos='n'
                elif tag.startswith('VB'): 
                    pos='v'
                else: 
                    pos='a'
                lemmatized_token=lemmatizer.lemmatize(each_token, pos)
                lemmatized_tokens.append(lemmatized_token)
            reconstructed_list.append(' '.join(lemmatized_tokens))
        return reconstructed_list
    
    def train_val_test_test2(self,df_train,df_test2,train_samp,test2_samp,
                             textvar,sentvar,maxlen):
        
        df_train_samp = df_train.sample(n=train_samp)
        df_test2_samp = df_test2.sample(n=test2_samp)
        X=df_train_samp[textvar]
        y=df_train_samp[sentvar]
        X_train, X_testval, y_train, y_testval=train_test_split(X, y, test_size=.3)
        X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, 
                                                        test_size=.33)
        
        # Fit and transform the data
        X_train=self.data_cleaning(X_train)
        X_val=self.data_cleaning(X_val)
        tokenizer=Tokenizer()
        tokenizer.fit_on_texts(X_train)
        vocab_size=len(tokenizer.word_index)+1
        X_train=pad_sequences(tokenizer.texts_to_sequences(X_train), 
                              maxlen=maxlen)
        X_val=pad_sequences(tokenizer.texts_to_sequences(X_val), 
                            maxlen=maxlen)
        
        # Transform the test data
        X_test=self.data_cleaning(X_test)
        X_test=pad_sequences(tokenizer.texts_to_sequences(X_test), 
                             maxlen=maxlen)
        
        X_test2=df_test2_samp[textvar]
        y_test2=df_test2_samp[sentvar]
        X_test2=self.data_cleaning(X_test2)
        X_test2=pad_sequences(tokenizer.texts_to_sequences(X_test2), 
                              maxlen=maxlen)
                
        return X_train,y_train,X_val,y_val,X_test,y_test,X_test2,y_test2,vocab_size
    
    def train_LSTM(self,vocab_size):
        model_reg=Sequential()
        model_reg.add(layers.Embedding(input_dim=vocab_size,\
                                   output_dim=100,\
                                   input_length=100))
        for i in range(hiddenlayers):
            if i+1 < hiddenlayers:
                model_reg.add(layers.Bidirectional(layers.LSTM(units=units, 
                                                               dropout=dropout, 
                                                    return_sequences=True)))
            else:
                model_reg.add(layers.Bidirectional(layers.LSTM(units=units, 
                                                               dropout=dropout)
                    ))        
        model_reg.add(layers.Dense(1))
        model_reg.compile(optimizer=Adam(learning_rate=learning_rate),\
                      loss='mse',\
                      metrics='mae')
        model_reg.fit(X_train,\
                  y_train,\
                  batch_size=256,\
                  epochs=epochs,\
                  validation_data=(X_val,y_val))
        model_reg_timestamp = str(datetime.now()).replace(' ','_').replace(':','')
        print(model_reg_timestamp)
    
    def hp_loop(self,n_units,dropouts,n_hiddenlayers,n_epochs,learning_rates):
        for units in n_units:
            for dropout in dropouts:
                for hiddenlayers in n_hiddenlayers:
                    for epochs in n_epochs:
                        for learning_rate in learning_rates:
                            print(units)
                            print(dropout)
                            print(hiddenlayers)
                            print(epochs)
                            print(learning_rate)
                            print('')
                            start = datetime.now()
                            self.train_LSTM(vocab_size)
                            