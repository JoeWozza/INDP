# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 11:34:42 2022

@author: Joe.WozniczkaWells
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from os import chdir, getcwd,listdir

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

import pandas as pd
import os

# Create folder in which to save visualisations
images_folder = '{0}/INDP/Images'.format(filepath)
audit_folder = '{0}/INDP/Images/VADER_audit'.format(filepath)

if not os.path.exists(images_folder):
    os.makedirs(images_folder)
if not os.path.exists(audit_folder):
    os.makedirs(audit_folder)
    
#%% Read in data

# Create input
filestring_eng = 'df_tweets_eng_tweepy_'

files = listdir("INDP/Data/Tweets")

tweets_files_eng = [s for s in files if filestring_eng in s]

df_tweets_eng = pd.DataFrame()

for f in tweets_files_eng:
    print(f)
    df_tweets_eng = df_tweets_eng.append(pd.read_csv("INDP/Data/Tweets/{0}".format(f)))

#pd.set_option('display.float_format', lambda x: '%.30f' % x)
df_tweets_eng['tweet_id'] = df_tweets_eng['tweet_id'].astype('Int64').apply(str)

df_tweets_deduped_eng = df_tweets_eng.drop_duplicates(subset=['tweet_id','search_term'])
df_tweets_deduped_eng = df_tweets_deduped_eng.set_index('tweet_id')

df_VADER = pd.read_csv("INDP/Data/VADER/df_VADER.csv")

# Categorise sentiment
def cat_sentiment(row):
    if row['sentiment'] >= 0.05:
        return 'Positive'
    elif row['sentiment'] <= - 0.05:
        return 'Negative'
    else:
        return 'Neutral'

df_VADER['sentiment_cat'] = df_VADER.apply(lambda row: cat_sentiment(row), axis=1)

# Join onto other columns
df_VADER['tweet_id'] = df_VADER['tweet_id'].astype('Int64').apply(str)
df_VADER = df_VADER.set_index('tweet_id')[['sentiment','sentiment_cat','sentconf']]
df_VADER = df_VADER.join(df_tweets_deduped_eng, how = 'right', lsuffix='_l',
                         rsuffix='_r')

# Dedupe by tweet_id
df_VADER_unique = df_VADER.reset_index().drop_duplicates(subset=['tweet_id']).set_index('tweet_id')

#%%

# Plot distribution of sentiment scores
fig, ax = plt.subplots(figsize = (12,6))
#fig = sns.kdeplot(data=df_VADER, x='sentiment', cut=0, ax=ax, color='#007C91')
fig = sns.distplot(df_VADER_unique['sentiment'], ax=ax, color='#007C91', kde_kws={'clip': (-1,1)})                
ax.set(xlabel = 'VADER sentiment score')
plt.tight_layout()
plt.savefig("{0}/VADER_sentiment.png".format(audit_folder))

# Plot distribution of sentiment confidence scores
fig, ax = plt.subplots(figsize = (12,6))
#fig = sns.kdeplot(data=df_VADER, x='sentconf', cut=0, ax=ax, color='#007C91')  
fig = sns.distplot(df_VADER_unique['sentconf'], ax=ax, color='#007C91', kde_kws={'clip': (0,1)})               
ax.set(xlabel = 'Confidence in VADER sentiment score')
plt.tight_layout()
plt.savefig("{0}/VADER_sentconf.png".format(audit_folder))

# Plot distribution of sentiment scores by search term
g = sns.FacetGrid(df_VADER, col = 'search_term', col_wrap = 5)
g.map(sns.histplot, 'sentiment', color='#007C91', kde=True, 
      kde_kws={'clip': (-1,1)}, linewidth=0)
g.set_axis_labels(x_var = 'VADER sentiment score', y_var = 'Density')
g.set_titles(col_template = '{col_name}')
g.tight_layout()
g.savefig("{0}/VADER_sentiment_search_term.png".format(audit_folder))

# Plot distribution of sentiment scores by search term
g = sns.FacetGrid(df_VADER, col = 'search_term', col_wrap = 5)
g.map(sns.histplot, 'sentconf', color='#007C91', kde=True, 
      kde_kws={'clip': (0,1)}, linewidth=0)
g.set_axis_labels(x_var = 'Confidence in VADER sentiment score', 
                  y_var = 'Density')
g.set_titles(col_template = '{col_name}')
g.tight_layout()
g.savefig("{0}/VADER_sentconf_search_term.png".format(audit_folder))

# Scatter plot of sentiment score v confidence score
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.scatterplot(data=df_VADER_unique, y='sentconf', x='sentiment', ax=ax, 
                      color='#007C91')
ax.set(ylabel = 'Confidence in VADER sentiment score', 
       xlabel = 'VADER sentiment score')
plt.tight_layout()
plt.savefig("{0}/VADER_sentiment_sentconf.png".format(audit_folder))

# Word clouds by sentiment_cat
# Wordcloud with stop words excluded
stopwords=set(STOPWORDS)
# Exclude a few other meaningless words
search_terms = df_VADER.drop_duplicates('search_term')['search_term'].tolist()
stopwords.update(['https','co','t','s','amp','u'])
stopwords.update(search_terms)

for cat in df_VADER_unique.drop_duplicates(['sentiment_cat'])['sentiment_cat'].tolist():
    print(cat)
    df = df_VADER_unique[df_VADER_unique['sentiment_cat']==cat]
    word_list = ' '.join(df['tweet_text'].tolist())
    
    wordcloud_excstopwords = WordCloud(stopwords=stopwords,
                                   background_color="white",
                                   max_words=len(word_list),max_font_size=40,
                                   relative_scaling=.5).generate(word_list)
    plt.figure()
    plt.imshow(wordcloud_excstopwords)
    plt.axis("off")
    plt.savefig('{0}/VADER_wordcloud_excstopwords_{1}.png'.format(audit_folder,cat))



