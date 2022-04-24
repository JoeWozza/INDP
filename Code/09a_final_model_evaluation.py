# -*- coding: utf-8 -*-
"""
This code is used to compare the output from the final model to the VADER
sentiment scores and evaluate the performance of the model.

@author: Joe Wozniczka-Wells
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
analyzer = SentimentIntensityAnalyzer()

from os import chdir

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import auc
from itertools import cycle

# Set file path
filepath = ("C:\\Users\\joew\\Documents\\Apprenticeship\\UoB\\SPFINDP21T4\\")
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
df_tweepy_LSTM_sent['tweet_date'] = pd.to_datetime(
        df_tweepy_LSTM_sent['tweet_date'])
df_tweepy_LSTM_sent_unique = df_tweepy_LSTM_sent.drop_duplicates(
        subset=['tweet_id'])

#%% Compare to VADER

# Define stopwords list
stopwords_fin = class_v.fin_stopwords(stopwords.words('english'))
# Sort out tweet_id field
df_tweepy_LSTM_sent_unique['tweet_id'] = (
        df_tweepy_LSTM_sent_unique['tweet_id'].astype('Int64').apply(str))
# Clean, tokenize and lemmatise tweet content
df_tweepy_LSTM_sent_unique['content_lemma'] = (
        df_tweepy_LSTM_sent_unique['tweet_text']
    .apply(class_v.clean)
    .apply(class_v.tokenize)
    .apply(class_v.lemmatise, stopwords_list=stopwords_fin))
# Calculate sentiment
df_tweepy_LSTM_sent_unique['VADER_sent'] = (
        df_tweepy_LSTM_sent_unique['content_lemma']
        .apply(class_v.VADERsentiment))
# Calculate sentiment confidence
df_tweepy_LSTM_sent_unique['VADER_conf'] = (
        df_tweepy_LSTM_sent_unique['content_lemma']
        .apply(class_v.sentconf))
# Categorise sentiment score
df_tweepy_LSTM_sent_unique['VADER_sent_cat'] = (df_tweepy_LSTM_sent_unique
                          .apply(lambda row: 
                              class_v.cat_sentiment(row['VADER_sent']), axis=1)
                              )
# Categorise sentiment confidence score
mean_cs,std_cs = class_v.cat_sentconf_stats(
        df_tweepy_LSTM_sent_unique['VADER_conf'])
df_tweepy_LSTM_sent_unique['VADER_conf_cat'] = (df_tweepy_LSTM_sent_unique
                          .apply(lambda row:
                              class_v.cat_sentconf(row['VADER_conf'],
                                                   mean_cs,std_cs), axis=1))

sentconf_cat_order = ['VeryHigh','High','Low','VeryLow','Zero']

# Compare to VADER scores

## Compare distributions
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.violinplot(data=df_tweepy_LSTM_sent_unique[['LSTM_sent',
                                                      'VADER_sent']],
                     cut=0,inner='quartile',ax=ax,color='#007C91')
ax.set_xticklabels(['LSTM','VADER'])
plt.suptitle('Comparison of LSTM and VADER distributions')
plt.tight_layout()
plt.savefig("{0}/dists_LSTM_VADER.png".format(finalmodel_folder))

## Compare LSTM scores/classes to VADER ones
fig, ax = plt.subplots(figsize = (12,6))
fig = sns.scatterplot(data=df_tweepy_LSTM_sent_unique,x='LSTM_sent',
                      y='VADER_sent',hue='VADER_conf_cat',
                      hue_order=sentconf_cat_order,ax=ax)
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

print(df_tweepy_LSTM_sent_unique
      .groupby('VADER_conf_cat')[['LSTM_sent','VADER_sent']].corr())

## Plot absolute error against sentconf
df_tweepy_LSTM_sent_unique['absolute_error'] = abs(
        df_tweepy_LSTM_sent_unique['LSTM_sent']-
        df_tweepy_LSTM_sent_unique['VADER_sent'])

fig, ax = plt.subplots(figsize = (12,6))
fig = sns.scatterplot(data=df_tweepy_LSTM_sent_unique, x='absolute_error', 
                      y='VADER_conf',color='#007C91')
ax.set(xlabel = 'Absolute difference between LSTM and VADER sentiment scores', 
       ylabel = 'Confidence in VADER score',
       title = 'Comparison of absolute error and VADER confidence')
plt.tight_layout()
plt.savefig("{0}/ae_v_sentconf.png".format(finalmodel_folder))

# Plot ROC curve (would be much easier with newer version of scikit-learn, 
# but can't install newest version on PHE laptop)
y_test = to_categorical(df_tweepy_LSTM_sent_unique.VADER_sent
                        .apply(class_v.cat_sentiment),num_classes=3)
y_score = to_categorical(df_tweepy_LSTM_sent_unique.LSTM_sent
                         .apply(class_v.cat_sentiment),num_classes=3)
n_classes=3
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
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
plt.plot(fpr["micro"],tpr["micro"],
         label="micro-average ROC curve (area = {0:0.2f})".format(
                 roc_auc["micro"]),color="deeppink",linestyle=":",linewidth=4)

plt.plot(fpr["macro"],tpr["macro"],
         label="macro-average ROC curve (area = {0:0.2f})".format(
                 roc_auc["macro"]),color="navy",linestyle=":",linewidth=4)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
cats = cycle(['neutral','positive','negative'])
for i, color, cat in zip(range(n_classes), colors, cats):
    plt.plot(fpr[i],tpr[i],color=color,lw=lw,
             label="ROC curve of class {0} (area = {1:0.2f})".format(
                     cat, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC curve")
plt.legend(loc="lower right")
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
plt.plot(fpr["micro"],tpr["micro"],
         label="micro-average ROC curve (area = {0:0.2f})".format(
                 roc_auc["micro"]),color="deeppink",linestyle=":",linewidth=4)

plt.plot(fpr["macro"],tpr["macro"],
         label="macro-average ROC curve (area = {0:0.2f})".format(
                 roc_auc["macro"]),color="navy",linestyle=":",linewidth=4)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
cats = cycle(['neutral','positive','negative'])
for i, color, cat in zip(range(n_classes), colors, cats):
    plt.plot(fpr[i],tpr[i],color=color,lw=lw,
             label="ROC curve of class {0} (area = {1:0.2f})".format(
                     cat, roc_auc[i]))

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC curve")
plt.legend(loc="lower right")
plt.savefig("{0}/ROC_curve_VeryHigh_High.png".format(finalmodel_folder))

# Check which category is which
print([sum(x) for x in zip(*y_score)])
print(pd.value_counts(df_tweepy_LSTM_sent_unique.LSTM_sent
                      .apply(class_v.cat_sentiment)))
print([sum(x) for x in zip(*y_test)])
print(pd.value_counts(df_tweepy_LSTM_sent_unique.VADER_sent
                      .apply(class_v.cat_sentiment)))

print(roc_auc_score(y_test,y_score))


