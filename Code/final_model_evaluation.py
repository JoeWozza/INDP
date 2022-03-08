# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:23:28 2022

@author: Joe.WozniczkaWells
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.ticker as tkr
from datetime import datetime
import numpy as np
import os
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from itertools import product

from os import chdir, listdir

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

from INDP.Code import LSTM
class_lstm = LSTM.LSTM()
from INDP.Code import VADER
class_v = VADER.VADER()

# Create folders
images_folder = 'INDP/Images'
finalmodel_folder = 'INDP/Images/Final_model'

if not os.path.exists(images_folder):
    os.makedirs(images_folder)
if not os.path.exists(finalmodel_folder):
    os.makedirs(finalmodel_folder)

df_tweepy_LSTM_sent = pd.read_csv("INDP/Data/LSTM/df_tweepy_LSTM_sent.csv")
# Categorise sentiment score
df_tweepy_LSTM_sent['LSTM_sent_cat'] = df_tweepy_LSTM_sent.apply(lambda row: 
    class_lstm.cat_sentiment_str(row['LSTM_sent']), axis=1)
# Convert tweet_date to datetime
df_tweepy_LSTM_sent['tweet_date'] = pd.to_datetime(df_tweepy_LSTM_sent['tweet_date'])
df_tweepy_LSTM_sent_unique = df_tweepy_LSTM_sent.drop_duplicates(subset=['tweet_id'])

#%% Compare to VADER

# Load packages
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
df_tweepy_LSTM_sent_unique['tweet_id'] = (df_tweepy_LSTM_sent_unique['tweet_id']
    .astype('Int64').apply(str))
# Clean, tokenize and lemmatise tweet content
df_tweepy_LSTM_sent_unique['content_lemma'] = (df_tweepy_LSTM_sent_unique['tweet_text']
    .apply(class_v.clean)
    .apply(class_v.tokenize)
    .apply(class_v.lemmatise, stopwords_list=stopwords_fin))
# Calculate sentiment
df_tweepy_LSTM_sent_unique['VADER_sent'] = (df_tweepy_LSTM_sent_unique['content_lemma']
    .apply(class_v.VADERsentiment))
# Calculate sentiment confidence
df_tweepy_LSTM_sent_unique['VADER_conf'] = (df_tweepy_LSTM_sent_unique['content_lemma']
    .apply(class_v.sentconf))
# Categorise sentiment score
df_tweepy_LSTM_sent_unique['VADER_sent_cat'] = df_tweepy_LSTM_sent_unique.apply(lambda row: 
    class_v.cat_sentiment(row['VADER_sent']), axis=1)
# Categorise sentiment confidence score
mean_cs,std_cs = class_v.cat_sentconf_stats(df_tweepy_LSTM_sent_unique['VADER_conf'])
df_tweepy_LSTM_sent_unique['VADER_conf_cat'] = df_tweepy_LSTM_sent_unique.apply(lambda row:
    class_v.cat_sentconf(row['VADER_conf'],mean_cs,std_cs), axis=1)

sentconf_cat_order = ['VeryHigh','High','Low','VeryLow','Zero']

# Compare to VADER scores

## Compare distributions
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.violinplot(data=df_tweepy_LSTM_sent_unique[['LSTM_sent','VADER_sent']],
                     cut=0,inner='quartile',ax=ax,color='#007C91')
ax.set_xticklabels(['LSTM','VADER'])
plt.suptitle('Comparison of LSTM and VADER distributions')
plt.tight_layout()
plt.savefig("{0}/dists_LSTM_VADER.png".format(finalmodel_folder))

## Compare LSTM scores/classes to VADER ones
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.scatterplot(data=df_tweepy_LSTM_sent_unique,x='LSTM_sent',y='VADER_sent', 
                      hue='VADER_conf_cat',hue_order=sentconf_cat_order, 
                      ax=ax)
ax.set(xlabel = 'LSTM sentiment score',ylabel = 'VADER sentiment score')
plt.legend(title='VADER confidence category',bbox_to_anchor=(1,1))
plt.suptitle('Comparison of LSTM predictions and VADER scores')
plt.tight_layout()
plt.savefig("{0}/LSTM_v_VADER.png".format(finalmodel_folder))

### What about by sentconf_cat
g = sns.relplot(data=df_tweepy_LSTM_sent_unique, x='LSTM_sent', y='VADER_sent', 
                col='VADER_conf_cat', col_order=sentconf_cat_order, col_wrap=2,
                color='#007C91')
g.set_axis_labels(x_var = 'LSTM sentiment score', 
                  y_var = 'VADER sentiment score')
g.set_titles(col_template = 'Confidence in VADER score: {col_name}')
g.fig.suptitle('Comparison of LSTM and VADER sentiment scores, by VADER '
               'confidence category')
g.tight_layout()
g.savefig("{0}/LSTM_reg_v_VADER_sentconf_cat.png".format(finalmodel_folder))

df_tweepy_LSTM_sent_unique.groupby('VADER_conf_cat')[['LSTM_sent','VADER_sent']].corr()

## Plot absolute error against sentconf
df_tweepy_LSTM_sent_unique['absolute_error'] = abs(df_tweepy_LSTM_sent_unique['LSTM_sent']-
                   df_tweepy_LSTM_sent_unique['VADER_sent'])

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.scatterplot(data=df_tweepy_LSTM_sent_unique, x='absolute_error', 
                      y='VADER_conf',color='#007C91')
ax.set(xlabel = 'Absolute difference between LSTM and VADER sentiment scores', 
       ylabel = 'Confidence in VADER score',
       title = 'Comparison of absolute error and VADER confidence')
plt.tight_layout()
plt.savefig("{0}/ae_v_sentconf.png".format(finalmodel_folder))

# Plot ROC curve (would be much easier with newer version of scikit-learn, but can't install newest version on PHE laptop)
from sklearn.metrics import roc_curve
from tensorflow.keras.utils import to_categorical

y_test = to_categorical(df_tweepy_LSTM_sent_unique.VADER_sent.apply(class_v.cat_sentiment),num_classes=3)
y_score = to_categorical(df_tweepy_LSTM_sent_unique.LSTM_sent.apply(class_v.cat_sentiment),num_classes=3)
n_classes=3
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
from sklearn.metrics import roc_curve, auc
from itertools import cycle
lw=2
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Macro-average ROC curve
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
cats = cycle(['neutral','positive','negative'])
for i, color, cat in zip(range(n_classes), colors, cats):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(cat, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC curve")
plt.legend(loc="lower right")
#plt.show()
plt.savefig("{0}/ROC_curve.png".format(finalmodel_folder))

# ROC curve for Very High/High confidence Tweets only
y_test = to_categorical(df_tweepy_LSTM_sent_unique[
        df_tweepy_LSTM_sent_unique['VADER_conf_cat'].isin(['VeryHigh','High'])]
    .VADER_sent.apply(class_v.cat_sentiment),num_classes=3)
y_score = to_categorical(df_tweepy_LSTM_sent_unique[
        df_tweepy_LSTM_sent_unique['VADER_conf_cat'].isin(['VeryHigh','High'])]
    .LSTM_sent.apply(class_v.cat_sentiment),num_classes=3)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Macro-average ROC curve
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
cats = cycle(['neutral','positive','negative'])
for i, color, cat in zip(range(n_classes), colors, cats):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(cat, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC curve")
plt.legend(loc="lower right")
#plt.show()
plt.savefig("{0}/ROC_curve_VeryHigh_High.png".format(finalmodel_folder))

# Check which category is which
print([sum(x) for x in zip(*y_score)])
print(pd.value_counts(df_tweepy_LSTM_sent_unique.LSTM_sent.apply(class_v.cat_sentiment)))
print([sum(x) for x in zip(*y_test)])
print(pd.value_counts(df_tweepy_LSTM_sent_unique.VADER_sent.apply(class_v.cat_sentiment)))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test,y_score))

#%% Investigate sentiment scores, overall, by search term and by UTLA

# Plot distribution of sentiment scores
fig, ax = plt.subplots(figsize = (12,6))
#fig = sns.kdeplot(data=df_VADER, x='sentiment', cut=0, ax=ax, color='#007C91')
fig = sns.distplot(df_tweepy_LSTM_sent_unique['LSTM_sent'], ax=ax, color='#007C91', 
                   kde_kws={'clip': (-1,1)})                
ax.set(xlabel = 'LSTM sentiment score')
plt.tight_layout()
plt.savefig("{0}/LSTM_tweepy_sentiment.png".format(finalmodel_folder))

# Plot distribution of sentiment scores by search term
search_terms = (df_tweepy_LSTM_sent_unique.drop_duplicates(subset=['search_term'])
    ['search_term'].sort_values())
g = sns.FacetGrid(df_tweepy_LSTM_sent_unique, col = 'search_term', col_wrap = 5, 
                  col_order = search_terms)
g.map(sns.histplot, 'LSTM_sent', color='#007C91', kde=True, 
      kde_kws={'clip': (-1,1)}, linewidth=0)
g.set_axis_labels(x_var = 'LSTM sentiment score', y_var = 'Density')
g.set_titles(col_template = '{col_name}')
g.tight_layout()
g.savefig("{0}/LSTM_tweepy_sentiment_search_term_hist.png".format(finalmodel_folder))

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.boxplot(data=df_tweepy_LSTM_sent_unique, x='search_term', y='LSTM_sent', 
                  color='#007C91', order = search_terms)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.savefig("{0}/LSTM_tweepy_sentiment_search_term_box.png".format(finalmodel_folder))

# Plot distribution of sentiment scores by UTLA
utlas = (df_tweepy_LSTM_sent.drop_duplicates(subset=['area'])
    ['area'].sort_values())
g = sns.FacetGrid(df_tweepy_LSTM_sent, col = 'area', col_wrap = 5,
                  col_order = utlas)
g.map(sns.histplot, 'LSTM_sent', color='#007C91', kde=True, 
      kde_kws={'clip': (-1,1)}, linewidth=0)
g.set_axis_labels(x_var = 'LSTM sentiment score', y_var = 'Density')
g.set_titles(col_template = '{col_name}')
g.tight_layout()
g.savefig("{0}/LSTM_tweepy_sentiment_UTLA_hist.png".format(finalmodel_folder))

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.boxplot(data=df_tweepy_LSTM_sent, x='area', y='LSTM_sent', 
                  color='#007C91', order = utlas)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.savefig("{0}/LSTM_tweepy_sentiment_UTLA_box.png".format(finalmodel_folder))

#%% Word clouds
# Wordcloud with stop words excluded
stopwords_wc=set(STOPWORDS)
# Exclude a few other meaningless words
stopwords_wc.update(['https','co','t','s','amp','u','covid'])
stopwords_wc.update(search_terms)

# By sentiment category
fig, ax = plt.subplots(1,3, figsize = (24,12))
for i,cat in enumerate(['Positive','Neutral','Negative']):
    df = df_tweepy_LSTM_sent_unique[df_tweepy_LSTM_sent_unique.LSTM_sent_cat==cat]
    word_list = ' '.join(df['tweet_text'].tolist())
    
    wordcloud_excstopwords = WordCloud(stopwords=stopwords_wc,
                                   background_color="white",
                                   max_words=len(word_list),max_font_size=40,
                                   relative_scaling=.5).generate(word_list)
    #plt.figure()
    ax[i].imshow(wordcloud_excstopwords)
    ax[i].axis("off")
    ax[i].set(title='LSTM sentiment: {0}'.format(cat))
fig.savefig('{0}/LSTM_tweepy_sent_wordclouds_sentcat.png'.format(finalmodel_folder))

# By UTLA
fig, ax = plt.subplots(6,4, figsize = (24,12))
for utla,ax_i in zip(utlas,ax.flatten()):
    df = df_tweepy_LSTM_sent[df_tweepy_LSTM_sent.area==utla]
    word_list = ' '.join(df['tweet_text'].tolist())
    
    wordcloud_excstopwords = WordCloud(stopwords=stopwords_wc,
                                   background_color="white",
                                   max_words=len(word_list),max_font_size=40,
                                   relative_scaling=.5).generate(word_list)
    #plt.figure()
    ax_i.imshow(wordcloud_excstopwords)
    ax_i.axis("off")
    ax_i.set(title='LSTM sentiment: {0}'.format(utla))
fig.savefig('{0}/LSTM_tweepy_sent_wordclouds_utla.png'.format(finalmodel_folder))

# By UTLA and sentiment category
for utla in utlas:
    fig, ax = plt.subplots(1,3, figsize = (16,12))
    for i,cat in enumerate(['Positive','Neutral','Negative']):
        df = df_tweepy_LSTM_sent[df_tweepy_LSTM_sent.LSTM_sent_cat==cat]
        word_list = ' '.join(df['tweet_text'].tolist())
        
        wordcloud_excstopwords = WordCloud(stopwords=stopwords_wc,
                                       background_color="white",
                                       max_words=len(word_list),
                                       max_font_size=40,
                                       relative_scaling=.5).generate(word_list)
        ax[i].imshow(wordcloud_excstopwords)
        ax[i].axis("off")
        ax[i].set(title='LSTM sentiment: {0}'.format(cat))
        fig.suptitle(utla, y=0.65)
        plt.tight_layout()
    fig.savefig('{0}/LSTM_tweepy_sent_wordclouds_sentcat_{1}.png'.format(
            finalmodel_folder,utla))

#%% Compare sentiment scores over time

#%% 7-day rolling averages

# Calculate rolling 7-day sentiment score
## Overall
df_tweepy_7day = (df_tweepy_LSTM_sent_unique.groupby('tweet_date')['LSTM_sent']
    .agg(['sum','count']).asfreq('d').reset_index())
df_tweepy_7day['sum'] = df_tweepy_7day['sum'].fillna(0)
df_tweepy_7day['count'] = df_tweepy_7day['count'].fillna(0)

df_tweepy_7day_rolling = df_tweepy_7day.rolling(window=7).sum()
df_tweepy_7day_rolling['mean'] = df_tweepy_7day_rolling['sum']/df_tweepy_7day_rolling['count']
df_tweepy_7day_rolling['mean'] = df_tweepy_7day_rolling['mean'].fillna(np.inf)

df_tweepy_7day_rolling = df_tweepy_7day[['tweet_date']].join(df_tweepy_7day_rolling)

all_dates = df_tweepy_7day.tweet_date
## By search term
dates_searchterms = pd.DataFrame(list(product(all_dates,search_terms)),
                                 columns=['tweet_date','search_term'])

df_tweepy_7day_search_term = (df_tweepy_LSTM_sent_unique.groupby(
        ['tweet_date','search_term'])['LSTM_sent']
    .agg(['sum','count']).reset_index().merge(dates_searchterms, 
        on=['tweet_date','search_term'], how='right'))
df_tweepy_7day_search_term['sum'] = df_tweepy_7day_search_term['sum'].fillna(0)
df_tweepy_7day_search_term['count'] = df_tweepy_7day_search_term['count'].fillna(0)

df_tweepy_7day_search_term['sum'] = df_tweepy_7day_search_term.groupby(['search_term'])['sum'].transform(lambda x: x.rolling(7).sum())
df_tweepy_7day_search_term['count'] = df_tweepy_7day_search_term.groupby(['search_term'])['count'].transform(lambda x: x.rolling(7).sum())

df_tweepy_7day_search_term['mean'] = df_tweepy_7day_search_term['sum']/df_tweepy_7day_search_term['count']
df_tweepy_7day_search_term['mean'] = df_tweepy_7day_search_term['mean'].fillna(np.inf)

## By UTLA
dates_tweepy_utlas = pd.DataFrame(list(product(all_dates,utlas)),
                                 columns=['tweet_date','area'])

df_tweepy_7day_utla = (df_tweepy_LSTM_sent.groupby(['tweet_date','area'])['LSTM_sent']
    .agg(['sum','count']).reset_index().merge(dates_tweepy_utlas, 
        on=['tweet_date','area'], how='right'))
df_tweepy_7day_utla['sum'] = df_tweepy_7day_utla['sum'].fillna(0)
df_tweepy_7day_utla['count'] = df_tweepy_7day_utla['count'].fillna(0)

df_tweepy_7day_utla['sum'] = df_tweepy_7day_utla.groupby(['area'])['sum'].transform(lambda x: x.rolling(7).sum())
df_tweepy_7day_utla['count'] = df_tweepy_7day_utla.groupby(['area'])['count'].transform(lambda x: x.rolling(7).sum())

df_tweepy_7day_utla['mean'] = df_tweepy_7day_utla['sum']/df_tweepy_7day_utla['count']
df_tweepy_7day_utla['mean'] = df_tweepy_7day_utla['mean'].fillna(np.inf)

# Plot overall 7-day average
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.lineplot(data=df_tweepy_7day_rolling, x='tweet_date', y='mean', ax=ax, 
                   color='#007C91')
ax.set(xlabel = 'Tweet date', ylabel = '7-day rolling average sentiment score')
plt.tight_layout()
plt.savefig("{0}/LSTM_tweepy_sentiment_roll7.png".format(finalmodel_folder))

# Plot 7-day average by search term
g = sns.relplot(data=df_tweepy_7day_search_term, x='tweet_date', y='mean', 
                col='search_term', col_wrap=5, kind='line', color='#007C91')
g.set_axis_labels(x_var = 'Tweet date', 
                  y_var = 'Mean sentiment score')
g.set_titles(col_template = 'Search term: {col_name}')
g.fig.suptitle('Sentiment scores over time, by search term')
g.tight_layout()
g.savefig("{0}/LSTM_tweepy_sentiment_roll7_searchterms.png".format(finalmodel_folder))

# Plot 7-day average by UTLA
g = sns.relplot(data=df_tweepy_7day_utla, x='tweet_date', y='mean', 
                col='area', col_wrap=5, kind='line', color='#007C91')
g.set_axis_labels(x_var = 'Tweet date', 
                  y_var = 'Mean sentiment score')
g.set_titles(col_template = '{col_name}')
g.fig.suptitle('Sentiment scores over time, by UTLA')
g.tight_layout()
g.savefig("{0}/LSTM_tweepy_sentiment_roll7_UTLAs.png".format(finalmodel_folder))

#%% Jan/Feb split

# Split into initial data (Jan only - used for training) and recent data 
# (15 Feb onwards)
def date_split(var):
    if var <= pd.to_datetime('2022-01-31'):
        return 'Initial'
    elif var >= pd.to_datetime('2022-02-15'):
        return 'Recent'

df_tweepy_LSTM_sent_unique['date_split'] = df_tweepy_LSTM_sent_unique['tweet_date'].apply(date_split)
df_tweepy_LSTM_sent_unique['month_year'] = pd.to_datetime(df_tweepy_LSTM_sent_unique['tweet_date']).dt.to_period('M')
df_tweepy_LSTM_sent_unique.groupby('month_year')['LSTM_sent'].mean()
df_tweepy_LSTM_sent_unique.groupby(['month_year','LSTM_sent_cat']).size()/df_tweepy_LSTM_sent_unique.groupby('month_year').size()

df_tweepy_LSTM_sent_unique_JanFeb = df_tweepy_LSTM_sent_unique[pd.to_datetime(df_tweepy_LSTM_sent_unique.tweet_date)>='2022-01-01']


df_tweepy_LSTM_sent['month_year'] = pd.to_datetime(df_tweepy_LSTM_sent['tweet_date']).dt.to_period('M')
df_tweepy_LSTM_sent_JanFeb = df_tweepy_LSTM_sent[pd.to_datetime(df_tweepy_LSTM_sent.tweet_date)>='2022-01-01']

# Overall box plot
fig, ax = plt.subplots(figsize=(12,6))
fig = sns.boxplot(data=df_tweepy_LSTM_sent_unique_JanFeb, x='month_year', y='LSTM_sent',
                  color='#007C91')
ax.set(xlabel = 'Tweet date', ylabel = 'Sentiment score', 
       title = 'Sentiment scores: January 2022 v February 2022')
plt.tight_layout()
plt.savefig("{0}/LSTM_tweepy_sentiment_JanFeb_box.png".format(finalmodel_folder))

# UTLA box plot
g = sns.catplot(data=df_tweepy_LSTM_sent_JanFeb, x='month_year', y='LSTM_sent', 
                col='area', col_wrap=5, kind='box', color='#007C91')
g.set_axis_labels(x_var = 'Tweet date', 
                  y_var = 'Sentiment score')
g.set_titles(col_template = '{col_name}')
g.fig.suptitle('Sentiment scores: January 2022 v February 2022, by UTLA')
g.tight_layout()
g.savefig("{0}/LSTM_tweepy_sentiment_JanFeb_UTLAs_box.png".format(finalmodel_folder))


#%% sntwitter all-pandemic data

df_snt_LSTM_sent = pd.read_csv("INDP/Data/LSTM/df_sntwitter_LSTM_sent.csv")
# Categorise sentiment score
df_snt_LSTM_sent['LSTM_sent_cat'] = df_snt_LSTM_sent.apply(lambda row: 
    class_lstm.cat_sentiment_str(row['LSTM_sent']), axis=1)
# Create date field from datetime
df_snt_LSTM_sent['tweet_date'] = pd.to_datetime(df_snt_LSTM_sent.tweet_datetime).dt.date
# Deduplicate by tweet_id
df_snt_LSTM_sent_unique = df_snt_LSTM_sent.drop_duplicates(subset=['tweet_id'])

#%% 7-day rolling averages

# Calculate rolling 7-day sentiment score
## Overall
df_snt_7day = (df_snt_LSTM_sent_unique.groupby('tweet_date')['LSTM_sent']
    .agg(['sum','count']).asfreq('d').reset_index())
df_snt_7day['sum'] = df_snt_7day['sum'].fillna(0)
df_snt_7day['count'] = df_snt_7day['count'].fillna(0)

df_snt_7day_rolling = df_snt_7day.rolling(window=7).sum()
df_snt_7day_rolling['mean'] = df_snt_7day_rolling['sum']/df_snt_7day_rolling['count']
df_snt_7day_rolling['mean'] = df_snt_7day_rolling['mean'].fillna(np.inf)

df_snt_7day_rolling = df_snt_7day[['tweet_date']].join(df_snt_7day_rolling)

all_dates = df_snt_7day.tweet_date
## By search term
dates_searchterms = pd.DataFrame(list(product(all_dates,search_terms)),
                                 columns=['tweet_date','search_term'])

df_snt_7day_search_term = (df_snt_LSTM_sent_unique.groupby(
        ['tweet_date','search_term'])['LSTM_sent']
    .agg(['sum','count']).reset_index().merge(dates_searchterms, 
        on=['tweet_date','search_term'], how='right'))
df_snt_7day_search_term['sum'] = df_snt_7day_search_term['sum'].fillna(0)
df_snt_7day_search_term['count'] = df_snt_7day_search_term['count'].fillna(0)

df_snt_7day_search_term['sum'] = df_snt_7day_search_term.groupby(['search_term'])['sum'].transform(lambda x: x.rolling(7).sum())
df_snt_7day_search_term['count'] = df_snt_7day_search_term.groupby(['search_term'])['count'].transform(lambda x: x.rolling(7).sum())

df_snt_7day_search_term['mean'] = df_snt_7day_search_term['sum']/df_snt_7day_search_term['count']
df_snt_7day_search_term['mean'] = df_snt_7day_search_term['mean'].fillna(np.inf)

## By UTLA
dates_snt_utlas = pd.DataFrame(list(product(all_dates,utlas)),
                                 columns=['tweet_date','area'])

df_snt_7day_utla = (df_snt_LSTM_sent.groupby(['tweet_date','area'])['LSTM_sent']
    .agg(['sum','count']).reset_index().merge(dates_snt_utlas, 
        on=['tweet_date','area'], how='right'))
df_snt_7day_utla['sum'] = df_snt_7day_utla['sum'].fillna(0)
df_snt_7day_utla['count'] = df_snt_7day_utla['count'].fillna(0)

df_snt_7day_utla['sum'] = df_snt_7day_utla.groupby(['area'])['sum'].transform(lambda x: x.rolling(7).sum())
df_snt_7day_utla['count'] = df_snt_7day_utla.groupby(['area'])['count'].transform(lambda x: x.rolling(7).sum())

df_snt_7day_utla['mean'] = df_snt_7day_utla['sum']/df_snt_7day_utla['count']
df_snt_7day_utla['mean'] = df_snt_7day_utla['mean'].fillna(np.inf)

# Plot overall 7-day average
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.lineplot(data=df_snt_7day_rolling, x='tweet_date', y='mean', ax=ax, 
                   color='#007C91')
ax.set(xlabel = 'Tweet date', ylabel = '7-day rolling average sentiment score')
plt.tight_layout()
plt.savefig("{0}/LSTM_snt_sentiment_roll7.png".format(finalmodel_folder))

# Plot 7-day average by search term
g = sns.relplot(data=df_snt_7day_search_term, x='tweet_date', y='mean', 
                col='search_term', col_wrap=5, kind='line', color='#007C91')
g.set_axis_labels(x_var = 'Tweet date', 
                  y_var = 'Mean sentiment score')
g.set_titles(col_template = 'Search term: {col_name}')
g.fig.suptitle('Sentiment scores over time, by search term')
g.tight_layout()
g.savefig("{0}/LSTM_snt_sentiment_roll7_searchterms.png".format(finalmodel_folder))

# Plot 7-day average by UTLA
g = sns.relplot(data=df_snt_7day_utla, x='tweet_date', y='mean', 
                col='area', col_wrap=5, kind='line', color='#007C91')
g.set_axis_labels(x_var = 'Tweet date', 
                  y_var = 'Mean sentiment score')
g.set_titles(col_template = '{col_name}')
g.fig.suptitle('Sentiment scores over time, by UTLA')
g.tight_layout()
g.savefig("{0}/LSTM_snt_sentiment_roll7_UTLAs.png".format(finalmodel_folder))











