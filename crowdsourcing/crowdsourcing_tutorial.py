# %% [markdown]
# # Crowdsourcing tutorial
# In this tutorial, we'll provide a simple walkthrough of how to use Snorkel to resolve conflicts
# in a noisy crowdsourced dataset for a sentiment analysis task.
# Like most Snorkel labeling pipelines, we'll use these denoised labels a deep learning model
# which can be applied to new, unseen data to automatically make predictions!
#
# In this tutorial, we're using the
# [Weather Sentiment](https://data.world/crowdflower/weather-sentiment)
# dataset from Figure Eight.
# In this task, contributors were asked to grade the sentiment of a particular tweet relating
# to the weather.
# Contributors could choose among the following categories:
#
# * Positive
# * Negative
# * I can't tell
# * Neutral / author is just sharing information
# * Tweet not related to weather condition
#
# The catch is that 20 contributors graded each tweet, and in many cases contributors assigned
# conflicting sentiment labels to the same tweet.
# This is a common issue when dealing with crowdsourced labeling workloads.
# Snorkel's ability to build high-quality datasets from multiple noisy labeling
# signals makes it an ideal framework to approach this problem.

# %%
from crowdsourcing.data import load_weather_data

crowd_labeled_tweets, gold_labels = load_weather_data()
crowd_labeled_tweets.head()

# %%
labels_by_annotator = crowd_labeled_tweets[
    ["tweet_id", "worker_id", "emotion"]
].groupby(["worker_id"])
labels_by_annotator.head()
