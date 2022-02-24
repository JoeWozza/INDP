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
import os
import numpy as np

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
    
    def train_val_test(self,df_train,train_samp,textvar,sentvar,maxlen):
        
        # Take samples of training dataset
        df_train_samp = df_train.sample(n=train_samp)
        # Split into X and y variables
        X=df_train_samp[textvar]
        y=df_train_samp[sentvar]
        # Do train/test split to get trainining set
        X_train, X_testval, y_train, y_testval=train_test_split(X, y, 
                                                                test_size=.3)
        # Do train/test split again to get test/validation sets
        X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, 
                                                        test_size=.33)
        
        # Fit and transform the data
        X_train=self.data_cleaning(X_train)
        tokenizer=Tokenizer()
        tokenizer.fit_on_texts(X_train)
        vocab_size=len(tokenizer.word_index)+1
        X_train=pad_sequences(tokenizer.texts_to_sequences(X_train), 
                              maxlen=maxlen)
        
        return X_train,y_train,X_val,y_val,X_test,y_test,vocab_size,tokenizer                
    
    def score_prep(self,X,tokenizer,maxlen):
        X=self.data_cleaning(X)
        X=pad_sequences(tokenizer.texts_to_sequences(X), maxlen=maxlen)
        return X
    
    def train_LSTM(self,X_train,y_train,X_val,y_val,vocab_size,units,dropout,
                   hiddenlayers,epochs,learning_rate):
        model=Sequential()
        model.add(layers.Embedding(input_dim=vocab_size,\
                                   output_dim=100,\
                                   input_length=100))
        for i in range(int(hiddenlayers)):
            if i+1 < hiddenlayers:
                model.add(layers.Bidirectional(layers.LSTM(units=units, 
                                                               dropout=dropout, 
                                                    return_sequences=True)))
            else:
                model.add(layers.Bidirectional(layers.LSTM(units=units, 
                                                               dropout=dropout)
                    ))        
        model.add(layers.Dense(1))
        model.compile(optimizer=Adam(learning_rate=learning_rate),\
                      loss='mse',\
                      metrics='mae')
        model.fit(X_train,\
                  y_train,\
                  batch_size=256,\
                  epochs=epochs,\
                  validation_data=(X_val,y_val))
        model_timestamp = str(datetime.now()).replace(' ','_').replace(':','')
        print(model_timestamp)
        return model, model_timestamp
        
    def save_model(self,model,tokenizer,model_timestamp,basefile):
        # Create folder for model
        if not os.path.exists('{0}/{1}'.format(basefile,model_timestamp)):
            os.makedirs('{0}/{1}'.format(basefile,model_timestamp))
        
        # Save model
        model.save('{0}/{1}/model.h5'.format(basefile,model_timestamp))
        # Save history
        np.save('{0}/{1}/model_history'.format(basefile,model_timestamp),
                model.history.history)
        
        # Save tokenizer
        with open('{0}/{1}/tokenizer.pickle'.format(basefile,model_timestamp),
                  'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def get_dataset(self,row):
        if "_" in row['variable']:
            val = 'valid'
        else:
            val = 'train'
        return val
            
    def epoch_perf_plot(self,model,model_timestamp,basefile):
        # Plot performance by number of epochs
        df_sns = pd.DataFrame(model.history.history)
        df_sns['epoch'] = df_sns.index + 1
        df_sns_melt = pd.melt(df_sns, id_vars='epoch')
        
        df_sns_melt['dataset'] = df_sns_melt.apply(lambda row: 
            self.get_dataset(row), axis=1)
        df_sns_melt['stat'] = df_sns_melt['variable'].str.split('_').str[-1]
        
        g = sns.relplot(data=df_sns_melt,x='epoch',y='value',hue='dataset', 
                        col='stat',kind='line')
        g.set_axis_labels(x_var = 'Epoch', y_var = 'Value')
        g.set_titles(col_template = '{col_name}')
        for ax in g.axes.flat:
            ax.xaxis.set_major_locator(tkr.MultipleLocator(5))
        g.tight_layout()
        g.savefig('{0}/{1}/stat_graphs.png'.format(basefile,model_timestamp))
        
    def test2_prep(self,df_test2,test2_samp,tokenizer,textvar,sentvar,maxlen):
        # Prepare test2 data
        df_test2_samp = df_test2.sample(n=test2_samp)
        X_test2=df_test2_samp[textvar]
        X_test2=self.score_prep(X_test2,tokenizer,maxlen)
        y_test2=df_test2_samp[sentvar]
        return X_test2,y_test2
    
    def score_dict(self,units,dropout,hiddenlayers,epochs,learning_rate,
                   start,end,y_test,predict_test,sentvar,y_test2,predict_test2,
                   basefile,model_timestamp):
        score_dict = {'model': model_timestamp,
                      'units': units,
                      'dropout': dropout,
                      'hiddenlayers': hiddenlayers,
                      'epochs': epochs,
                      'learning_rate': learning_rate,
                      'time_taken': str(end-start),
                      'test_mse':mean_squared_error(y_test,predict_test),
                      'test_mae':median_absolute_error(y_test,predict_test),
                      'test_corr':y_test.reset_index()[sentvar].corr(
                              pd.Series(predict_test[:,0])),
                      'test2_mse':mean_squared_error(y_test2,predict_test2),
                      'test2_mae':median_absolute_error(y_test2,predict_test2),
                      'test2_corr':y_test2.reset_index()[sentvar].corr(
                              pd.Series(predict_test2[:,0]))
                          }
        np.save('{0}/{1}/score_dict'.format(basefile,model_timestamp),
                score_dict)
        return score_dict                    
    
    def hp_loop(self,df_train,df_hp,basefile,train_samp,textvar,sentvar,
                maxlen,df_test2,test2_samp):
        X_train,y_train,X_val,y_val,X_test,y_test,vocab_size,tokenizer = (
                self.train_val_test(df_train,train_samp,textvar,sentvar,maxlen)
                )
        # Prepare validation data
        X_val=self.score_prep(X_val,tokenizer,maxlen)
        # Prepare test data
        X_test=self.score_prep(X_test,tokenizer,maxlen)
        df_scores=pd.DataFrame()
        
        for index,row in df_hp.iterrows():
            
            dropout = row['dropout']
            epochs = row['epochs'].astype('int')
            hiddenlayers = row['hiddenlayers']
            learning_rate = row['learning_rate']
            units = row['units'].astype('int')
            
            print(dropout)
            print(epochs)
            print(hiddenlayers)
            print(learning_rate)
            print(units)
            print('')
            
            start = datetime.now()
            # Train LSTM model
            model,model_timestamp = self.train_LSTM(X_train,y_train,X_val,
                                                    y_val,vocab_size,units,
                                                    dropout,hiddenlayers,
                                                    epochs,learning_rate)
            # Save LSTM model and related history and tokenizer
            self.save_model(model,tokenizer,model_timestamp,basefile)
            # Plot performance by number of epochs
            self.epoch_perf_plot(model,model_timestamp,basefile)
            # Prepare test2 data
            X_test2,y_test2=self.test2_prep(df_test2,test2_samp,tokenizer,
                                            textvar,sentvar,maxlen)
            # Score model on test dataset
            predict_test = model.predict(X_test)
            # Score model on 'test2' dataset                
            predict_test2 = model.predict(X_test2)
            # Print time taken
            end = datetime.now()
            print(str(end-start))
            # Output performance metrics
            score_dict = self.score_dict(units,dropout,hiddenlayers,epochs,
                                         learning_rate,start,end,y_test,
                                         predict_test,sentvar,y_test2,
                                         predict_test2,basefile,
                                         model_timestamp)
            
            df_scores = df_scores.append(score_dict,ignore_index=True)
        return df_scores
                            
                            