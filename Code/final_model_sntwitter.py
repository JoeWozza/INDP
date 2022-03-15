# -*- coding: utf-8 -*-
"""
This code is used to investigate the output from the final model on the Tweets
downloaded using sntwitter. This includes producing visualisations for the 
final report.

@author: Joe Wozniczka-Wells
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

from os import chdir

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

#%% sntwitter all-pandemic data

df_snt_LSTM_sent = pd.read_csv("INDP/Data/LSTM/df_sntwitter_LSTM_sent.csv")
# Categorise sentiment score
df_snt_LSTM_sent['LSTM_sent_cat'] = df_snt_LSTM_sent.apply(lambda row: 
    class_lstm.cat_sentiment_str(row['LSTM_sent']), axis=1)
# Create date field from datetime
df_snt_LSTM_sent['tweet_date'] = pd.to_datetime(pd.to_datetime(
        df_snt_LSTM_sent.tweet_datetime).dt.date)
# Deduplicate by tweet_id
df_snt_LSTM_sent_unique = df_snt_LSTM_sent.drop_duplicates(subset=['tweet_id'])

search_terms = (df_snt_LSTM_sent_unique.drop_duplicates(
        subset=['search_term'])['search_term'].sort_values())
utlas = (df_snt_LSTM_sent_unique.drop_duplicates(subset=['area'])
    ['area'].sort_values())
cats = (df_snt_LSTM_sent_unique.drop_duplicates(subset=['LSTM_sent_cat'])
    ['LSTM_sent_cat'].sort_values())

#%% Rolling averages
df_snt_LSTM_sent_unique.to_csv('INDP/Data/LSTM/test.csv')
# Calculate rolling sentiment score
# Define window length (days)
wind=30
## Overall
df_snt_day = (df_snt_LSTM_sent_unique.groupby('tweet_date')['LSTM_sent']
    .agg(['sum','count']).asfreq('d').reset_index())
df_snt_day['sum'] = df_snt_day['sum'].fillna(0)
df_snt_day['count'] = df_snt_day['count'].fillna(0)

df_snt_rolling = df_snt_day.rolling(window=wind).sum()
df_snt_rolling['mean'] = (df_snt_rolling['sum']/df_snt_rolling['count'])
df_snt_rolling['mean'] = df_snt_rolling['mean'].fillna(np.inf)

df_snt_rolling = df_snt_day[['tweet_date']].join(df_snt_rolling)

all_dates = df_snt_day.tweet_date
## By search term
dates_searchterms = pd.DataFrame(list(product(all_dates,search_terms)),
                                 columns=['tweet_date','search_term'])

df_snt_day_search_term = (df_snt_LSTM_sent_unique.groupby(
        ['tweet_date','search_term'])['LSTM_sent']
    .agg(['sum','count']).reset_index().merge(dates_searchterms, 
        on=['tweet_date','search_term'], how='right'))
df_snt_day_search_term['sum'] = df_snt_day_search_term['sum'].fillna(0)
df_snt_day_search_term['count'] = df_snt_day_search_term['count'].fillna(0)

df_snt_day_search_term['sum'] = (df_snt_day_search_term
                       .groupby(['search_term'])['sum']
                       .transform(lambda x: x.rolling(wind).sum()))
df_snt_day_search_term['count'] = (df_snt_day_search_term
                       .groupby(['search_term'])['count']
                       .transform(lambda x: x.rolling(wind).sum()))

df_snt_day_search_term['mean'] = (df_snt_day_search_term['sum']/
                       df_snt_day_search_term['count'])
df_snt_day_search_term['mean'] = (df_snt_day_search_term['mean'].
                       fillna(np.inf))

## By UTLA
dates_snt_utlas = pd.DataFrame(list(product(all_dates,utlas)),
                                 columns=['tweet_date','area'])

df_snt_day_utla = (df_snt_LSTM_sent
                   .groupby(['tweet_date','area'])['LSTM_sent']
                   .agg(['sum','count']).reset_index().merge(dates_snt_utlas,
                       on=['tweet_date','area'], how='right'))
df_snt_day_utla['sum'] = df_snt_day_utla['sum'].fillna(0)
df_snt_day_utla['count'] = df_snt_day_utla['count'].fillna(0)

df_snt_day_utla['sum'] = (df_snt_day_utla
               .groupby(['area'])['sum']
               .transform(lambda x: x.rolling(wind).sum()))
df_snt_day_utla['count'] = (df_snt_day_utla
               .groupby(['area'])['count']
               .transform(lambda x: x.rolling(wind).sum()))

df_snt_day_utla['mean'] = df_snt_day_utla['sum']/df_snt_day_utla['count']
df_snt_day_utla['mean'] = df_snt_day_utla['mean'].fillna(np.inf)

# Plot overall 7-day average
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.lineplot(data=df_snt_rolling, x='tweet_date', y='mean', ax=ax, 
                   color='#007C91')
ax.set(xlabel = 'Tweet date',
       ylabel = '{wind}-day rolling average sentiment score')
ax.axvline(pd.to_datetime('2020-11-02'), linestyle='solid', color='#00AB8E',
           label='Pfizer vaccine approved by MHRA')
ax.axvline(pd.to_datetime('2020-12-08'), linestyle='dashed', color='#00AB8E',
           label='First COVID-19 vaccine administered by NHS')
ax.axvline(pd.to_datetime('2021-04-07'), linestyle='solid', color='#E40046',
           label='Under 30s offered alternative to Oxford/AZ vaccine, due to '
           'blood clot links')
ax.axvline(pd.to_datetime('2021-05-07'), linestyle='dashed', color='#E40046',
           label='Under 40s offered alternative to Oxford/AZ vaccine, due to '
           'blood clot links')
ax.axvline(pd.to_datetime('2021-07-12'), linestyle='dashdot', color='#00AB8E',
           label='80% of over 12s in UK receive at least one vaccine dose')
ax.axvline(pd.to_datetime('2021-11-12'), linestyle='dotted', color='#00AB8E',
           label='80% of over 12s in UK receive at least two vaccine doses')
ax.axvline(pd.to_datetime('2021-11-27'), linestyle='dashdot', color='#E40046',
           label='First case of Omicron variant identified in UK')
ax.legend(loc=(1.05,0))
plt.tight_layout()
plt.savefig("{0}/LSTM_snt_sentiment_roll{1}.png".format(finalmodel_folder,
            wind))

# Plot rolling average by search term
g = sns.relplot(data=df_snt_day_search_term, x='tweet_date', y='mean', 
                col='search_term', col_wrap=5, kind='line', color='#007C91')
g.set_axis_labels(x_var = 'Tweet date', 
                  y_var = 'Mean sentiment score')
g.set_titles(col_template = 'Search term: {col_name}')
g.fig.suptitle('Sentiment scores over time, by search term')
g.tight_layout()
g.savefig("{0}/LSTM_snt_sentiment_roll{1}_searchterms.png".format(
        finalmodel_folder,wind))

# Plot 7-day average by UTLA
df_snt_day_utla['area_type'] = 'UTLA'
df_snt_rolling['area_type'] = 'National'

for utla in utlas:
    df_snt_rolling['area'] = utla
    df_snt_day_utla = df_snt_day_utla.append(df_snt_rolling)

g = sns.relplot(data=df_snt_day_utla, x='tweet_date', y='mean', col='area', 
                col_wrap=5, kind='line', palette=['#007C91','#582C83'], 
                                                  hue='area_type')
g.set_axis_labels(x_var = 'Tweet date', 
                  y_var = '{wind}-day rolling average sentiment score')
g.set_titles(col_template = '{col_name}')
g.set_xticklabels(rotation=30)
for i,ax in enumerate(g.axes.flat):
    ax.axvline(pd.to_datetime('2020-11-02'), linestyle='solid', 
               color='#00AB8E', label='Pfizer vaccine approved by MHRA')
    ax.axvline(pd.to_datetime('2020-12-08'), linestyle='dashed', 
               color='#00AB8E',
               label='First COVID-19 vaccine administered by NHS')
    ax.axvline(pd.to_datetime('2021-04-07'), linestyle='solid', 
               color='#E40046',
               label='Under 30s offered alternative to Oxford/AZ vaccine, due '
               'to blood clot links')
    ax.axvline(pd.to_datetime('2021-05-07'), linestyle='dashed', 
               color='#E40046',
               label='Under 40s offered alternative to Oxford/AZ vaccine, due '
               'to blood clot links')
    ax.axvline(pd.to_datetime('2021-07-12'), linestyle='dashdot', 
               color='#00AB8E',
               label='80% of over 12s in UK receive at least one vaccine dose')
    ax.axvline(pd.to_datetime('2021-11-12'), linestyle='dotted', 
               color='#00AB8E',
               label='80% of over 12s in UK receive at least two vaccine doses'
               )
    ax.axvline(pd.to_datetime('2021-11-27'), linestyle='dashdot', 
               color='#E40046',
               label='First case of Omicron variant identified in UK')
    if i == len(g.axes.flat)-1:
        ax.legend(loc=(-3,-0.5))
g._legend.set_title("Average")
g.tight_layout()
g.savefig("{0}/LSTM_snt_sentiment_roll{1}_UTLAs.png".format(finalmodel_folder,
          wind))

#%% Analyse percentage of positive/negative/neutral tweets over time

# Overall
dates_cats = pd.DataFrame(list(product(all_dates,cats)),
                          columns=['tweet_date','LSTM_sent_cat'])

df_snt_day_cat = (df_snt_LSTM_sent_unique
                  .groupby(['tweet_date','LSTM_sent_cat']).size()
                  .to_frame('date_cat_tot')
                  .reset_index().merge(dates_cats,
                              on=['tweet_date','LSTM_sent_cat'], how='right')
                  .set_index('tweet_date'))
df_snt_day_cat['date_cat_tot'] = df_snt_day_cat['date_cat_tot'].fillna(0)
df_snt_day_cat['date_tot'] = df_snt_day_cat.groupby('tweet_date').agg('sum')
df_snt_day_cat['date_cat_prop'] = (df_snt_day_cat.date_cat_tot/
              df_snt_day_cat.date_tot)

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.lineplot(data=df_snt_day_cat, x='tweet_date', y='date_cat_prop', 
                   ax=ax, color='#007C91', hue='LSTM_sent_cat')
ax.set(xlabel = 'Tweet date',
       ylabel = 'Proportion of tweets in sentiment category')
ax.axvline(pd.to_datetime('2020-11-02'), linestyle='solid', color='#00AB8E',
           label='Pfizer vaccine approved by MHRA')
ax.axvline(pd.to_datetime('2020-12-08'), linestyle='dashed', color='#00AB8E',
           label='First COVID-19 vaccine administered by NHS')
ax.axvline(pd.to_datetime('2021-04-07'), linestyle='solid', color='#E40046',
           label='Under 30s offered alternative to Oxford/AZ vaccine, due to '
           'blood clot links')
ax.axvline(pd.to_datetime('2021-05-07'), linestyle='dashed', color='#E40046',
           label='Under 40s offered alternative to Oxford/AZ vaccine, due to '
           'blood clot links')
ax.axvline(pd.to_datetime('2021-07-12'), linestyle='dashdot', color='#00AB8E',
           label='80% of over 12s in UK receive at least one vaccine dose')
ax.axvline(pd.to_datetime('2021-11-12'), linestyle='dotted', color='#00AB8E',
           label='80% of over 12s in UK receive at least two vaccine doses')
ax.axvline(pd.to_datetime('2021-11-27'), linestyle='dashdot', color='#E40046',
           label='First case of Omicron variant identified in UK')
ax.legend(loc=(1.05,0))
plt.tight_layout()
# Some patterns visible but huge variation in the plot above. Calculate rolling
# proportions by category.

df_snt_day_cat_roll = df_snt_day_cat.copy().reset_index()

date_cat_tot=(df_snt_day_cat_roll
              .groupby(['LSTM_sent_cat'])
              .rolling(30,on='tweet_date')[['date_cat_tot','tweet_date']]
              .sum())

date_tot=(df_snt_day_cat_roll
              .drop_duplicates('tweet_date')
              .rolling(30,on='tweet_date')[['date_tot','tweet_date']]
              .sum()
              .set_index('tweet_date'))

df_snt_day_cat_roll=df_snt_day_cat_roll.set_index('tweet_date')
df_snt_day_cat_roll['date_tot']=date_tot
df_snt_day_cat_roll=(df_snt_day_cat_roll.reset_index().set_index([
        'LSTM_sent_cat','tweet_date']))
df_snt_day_cat_roll['date_cat_tot']=date_cat_tot
df_snt_day_cat_roll['date_cat_prop'] = (df_snt_day_cat_roll.date_cat_tot/
              df_snt_day_cat_roll.date_tot)

# Plot rolling average
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.lineplot(data=df_snt_day_cat_roll, x='tweet_date', y='date_cat_prop', 
                   ax=ax, palette=['#007C91','#FF7F32','#E40046'], 
                                   hue='LSTM_sent_cat')
ax.set(xlabel = 'Tweet date',
       ylabel = 'Proportion of tweets in sentiment category')
ax.axvline(pd.to_datetime('2020-11-02'), linestyle='solid', color='#00AB8E',
           label='Pfizer vaccine approved by MHRA')
ax.axvline(pd.to_datetime('2020-12-08'), linestyle='dashed', color='#00AB8E',
           label='First COVID-19 vaccine administered by NHS')
ax.axvline(pd.to_datetime('2021-04-07'), linestyle='solid', color='#E40046',
           label='Under 30s offered alternative to Oxford/AZ vaccine, due to '
           'blood clot links')
ax.axvline(pd.to_datetime('2021-05-07'), linestyle='dashed', color='#E40046',
           label='Under 40s offered alternative to Oxford/AZ vaccine, due to '
           'blood clot links')
ax.axvline(pd.to_datetime('2021-07-12'), linestyle='dashdot', color='#00AB8E',
           label='80% of over 12s in UK receive at least one vaccine dose')
ax.axvline(pd.to_datetime('2021-11-12'), linestyle='dotted', color='#00AB8E',
           label='80% of over 12s in UK receive at least two vaccine doses')
ax.axvline(pd.to_datetime('2021-11-27'), linestyle='dashdot', color='#E40046',
           label='First case of Omicron variant identified in UK')
ax.legend(loc=(1.05,0))
plt.tight_layout()
plt.savefig("{0}/LSTM_snt_sentcat_roll{1}.png".format(finalmodel_folder,
          wind))

# Calculate rolling proportions by UTLA
dates_cats_utlas = pd.DataFrame(list(product(all_dates,cats,utlas)),
                                columns=['tweet_date','LSTM_sent_cat','area'])

df_snt_day_cat_utla = (df_snt_LSTM_sent_unique
                  .groupby(['area','tweet_date','LSTM_sent_cat']).size()
                  .to_frame('date_cat_utla_tot')
                  .reset_index().merge(dates_cats_utlas,
                              on=['tweet_date','LSTM_sent_cat','area'], 
                              how='right').set_index(['area','tweet_date']))
df_snt_day_cat_utla['date_cat_utla_tot'] = (
        df_snt_day_cat_utla['date_cat_utla_tot'].fillna(0))
df_snt_day_cat_utla['date_utla_tot'] = (df_snt_day_cat_utla
                   .groupby(['area','tweet_date']).agg('sum'))

df_snt_day_cat_utla_roll = df_snt_day_cat_utla.copy().reset_index()

date_cat_utla_tot=(df_snt_day_cat_utla_roll
              .groupby(['area','LSTM_sent_cat'])
              .rolling(30,on='tweet_date')[['date_cat_utla_tot','tweet_date']]
              .sum())

date_utla_tot=(df_snt_day_cat_utla_roll
               .drop_duplicates(['area','tweet_date'])
               .groupby('area')
               .rolling(30,on='tweet_date')[['date_utla_tot','tweet_date']]
               .sum())

df_snt_day_cat_utla_roll=df_snt_day_cat_utla_roll.set_index(['area',
                                                             'tweet_date'])
df_snt_day_cat_utla_roll['date_utla_tot']=date_utla_tot
df_snt_day_cat_utla_roll=(df_snt_day_cat_utla_roll.reset_index().set_index([
        'area','LSTM_sent_cat','tweet_date']))
df_snt_day_cat_utla_roll['date_cat_utla_tot']=date_cat_utla_tot
df_snt_day_cat_utla_roll['date_cat_utla_prop'] = (
        df_snt_day_cat_utla_roll.date_cat_utla_tot/
        df_snt_day_cat_utla_roll.date_utla_tot)

df_snt_day_cat_utla_roll=df_snt_day_cat_utla_roll.reset_index()

# Plot rolling average by UTLA
g = sns.relplot(data=df_snt_day_cat_utla_roll, x='tweet_date', 
                y='date_cat_utla_prop', col='area', col_wrap=5, kind='line', 
                palette=['#007C91','#FF7F32','#E40046'], hue='LSTM_sent_cat')
g.set_axis_labels(x_var = 'Tweet date', 
                  y_var = '{0}-day rolling average sentiment score'.format(
                          wind))
g.set_titles(col_template = '{col_name}')
g.set_xticklabels(rotation=30)
for i,ax in enumerate(g.axes.flat):
    ax.axvline(pd.to_datetime('2020-11-02'), linestyle='solid', 
               color='#00AB8E', label='Pfizer vaccine approved by MHRA')
    ax.axvline(pd.to_datetime('2020-12-08'), linestyle='dashed', 
               color='#00AB8E',
               label='First COVID-19 vaccine administered by NHS')
    ax.axvline(pd.to_datetime('2021-04-07'), linestyle='solid', 
               color='#E40046',
               label='Under 30s offered alternative to Oxford/AZ vaccine, due '
               'to blood clot links')
    ax.axvline(pd.to_datetime('2021-05-07'), linestyle='dashed', 
               color='#E40046',
               label='Under 40s offered alternative to Oxford/AZ vaccine, due '
               'to blood clot links')
    ax.axvline(pd.to_datetime('2021-07-12'), linestyle='dashdot', 
               color='#00AB8E',
               label='80% of over 12s in UK receive at least one vaccine dose')
    ax.axvline(pd.to_datetime('2021-11-12'), linestyle='dotted', 
               color='#00AB8E',
               label='80% of over 12s in UK receive at least two vaccine doses'
               )
    ax.axvline(pd.to_datetime('2021-11-27'), linestyle='dashdot', 
               color='#E40046',
               label='First case of Omicron variant identified in UK')
    if i == len(g.axes.flat)-1:
        ax.legend(loc=(-3,-0.5))
g._legend.set_title("Sentiment category")
g.tight_layout()
g.savefig("{0}/LSTM_snt_sentcat_roll{1}_UTLAs.png".format(finalmodel_folder,
          wind))
