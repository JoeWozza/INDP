# INDP

The code in this repository can be used to carry out semi-supervised sentiment analysis on Tweets within user-selected UK upper-tier local authorities, downloaded using keyword searches of the users choice. VADER is used to produce pseudo-labels of the sentiment of the Tweets, some of which are then used to train an LSTM model to improve the sentiment predictions. This model is then scored out on the Tweets and sentiment is analysed using a variety of visualisation methods.

The scripts should be run in the following order:

1. 01_circles.py: This script uses the CircleApprox class from CircleApprox.py to produce approximations of geographical areas using overlapping circles.
2. 02a_vaccine_scrape_tweepy.py: This script uses the TweetScrape class from TweepyScrape.py to download Tweets from the last 9 days.
3. 02b_vaccine_scrape_sntwitter.py: This script uses the TweetScrape class from sntwitterScrape.py to download Tweets over a longer period of time.
4. 03a_tweet_audit_England.py: This code is used to audit the England Tweets dataset.
5. 03b_tweet_audit_Midlands.py: This code is used to audit the Midlands Tweets dataset.
6. 04_clean_tweets.py: This code cleans the data downloaded from Twitter.
7. 05_lexicon.py: This code uses the VADER class to get sentiment scores (-1 to 1) and sentiment confidence scores (0 to 1) for Tweets captured in several dated csv files.
8. 06_lex_audit.py: This code is used to investigate the output from Code/05_lexicon.py.
9. 07_supervised_hyperparametertuning.py: This code is used to perform two rounds of hyperparameter tuning on the LSTM model and select the best model.
10. 08_supervised_score.py: This script scores the final model on the TweePy and sntwitter data.
11. 09a_final_model_evaluation.py: This code is used to compare the output from the final model to the VADER sentiment scores and evaluate the performance of the model.
12. 09b_final_model_tweepy.py: This code is used to investigate the output from the final model on the Tweets downloaded using TweePy. This includes producing visualisations for the final report.
13. 09c_final_model_sntwitter.py: This code is used to investigate the output from the final model on the Tweets downloaded using sntwitter. This includes producing visualisations for the  final report.

The scripts whose names begin with 'XX_' were used on an ad-hoc basis in the creation of this project and are included for context. The remaining scripts contain classes and functions that are used in the numbered scripts.
