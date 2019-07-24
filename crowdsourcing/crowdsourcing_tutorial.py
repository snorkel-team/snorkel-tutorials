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

# %% [markdown]
# We start by loading our data. It has 632 examples, which are divided into development, validation, and test sets of size 50 each, and a training set with the remaining 482 examples.

# %%
import os

if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("crowdsourcing")
from data import load_data, emotions

print(f"Emotion to int mapping: {emotions}")

crowd_answers, (
    (dev_df, dev_labels),
    (val_df, val_labels),
    (test_df, test_labels),
    train_df,
) = load_data()
crowd_answers.head()

# %% [markdown]
# ## Writing Labeling Functions
# Each crowd worker can be thought of as a single labeling function, as each worker labels a subset of examples, and may have errors or conflicting answers with other workers / labeling functions. So we create one labeling function per worker.

# %%
labels_by_annotator = crowd_answers[["tweet_id", "worker_id", "emotion"]].groupby(
    ["worker_id"]
)
worker_dicts = {}
for worker_id in labels_by_annotator.groups:
    worker_df = labels_by_annotator.get_group(worker_id)[["tweet_id", "emotion"]]
    worker_dicts[worker_id] = dict(zip(worker_df.tweet_id, worker_df.emotion))

# %%
from snorkel.labeling.lf import LabelingFunction, labeling_function


def get_worker_labeling_function(worker_id):
    worker_dict = worker_dicts[worker_id]

    def f(x, worker_dict):
        return worker_dict.get(x.tweet_id, -1)

    return LabelingFunction(
        f"worker_{worker_id}", f=f, resources={"worker_dict": worker_dict}
    )


worker_lfs = [get_worker_labeling_function(worker_id) for worker_id in worker_dicts]

# %% [markdown]
# ### Additional labeling functions
# We can mix the 'crowd worker' labeling functions with labeling functions of other types, and have the label model learn how to combine their values.
#
# For example, the [TextBlob](https://textblob.readthedocs.io/en/dev/index.html) tool provides a pretrained sentiment analyzer. We can use it's sentiment scores to create more labeling functions.

# %%
import matplotlib.pyplot as plt
from textblob import TextBlob
import statistics

polarities_list = []
for j in range(4):
    polarities_list.append(
        [
            TextBlob(x.tweet_text).sentiment.polarity
            for (i, x), label in zip(dev_df.iterrows(), dev_labels)
            if label == j
        ]
    )

print(f"Median polarity for label 0: {statistics.median(polarities_list[0])}")
print(f"Median polarity for label 1: {statistics.median(polarities_list[1])}")
print(f"Median polarity for label 2: {statistics.median(polarities_list[2])}")
print(f"Median polarity for label 3: {statistics.median(polarities_list[3])}")
_ = plt.hist(polarities_list)


# %% [markdown]
# We observe two things: Label 3 (red) often has a polarity close to 0. And label 0 (blue) often has a high positive polarity (median 0.483). We can convert these two observations into two corresponding labeling functions.
#

# %%
@labeling_function()
def positive_polarity(x):
    return 0 if TextBlob(x.tweet_text).sentiment.polarity > 0.3 else -1


@labeling_function()
def low_absolute_polarity(x):
    return 2 if abs(TextBlob(x.tweet_text).sentiment.polarity) < 0.1 else -1


# %% [markdown]
# ### Apply Labeling Functions

# %%
from snorkel.labeling.apply import PandasLFApplier

lfs = worker_lfs + [positive_polarity, low_absolute_polarity]

applier = PandasLFApplier(lfs)
dev_L = applier.apply(dev_df)
train_L = applier.apply(train_df)

# %%
# Analyse labeling functions on dev set.
from snorkel.labeling.analysis import LFAnalysis

lf_names = [lf.name for lf in lfs]
LFAnalysis(dev_L).lf_summary(dev_labels, lf_names=lf_names)

# %% [markdown]
# ## Train Label Model And Generate Soft Labels

# %%
from snorkel.labeling.model.label_model import LabelModel

# Count frequency of labels in dev set to provide prior to label model.
import collections

counter = collections.Counter(dev_labels)
class_balance = [
    (counter[label] + 1.0) / (sum(counter.values()) + 5.0) for label in range(5)
]
print("Estimated priors for labels: {class_balance}")

# Train label model.
label_model = LabelModel(cardinality=5, verbose=True)
label_model.fit(
    train_L,
    n_epochs=5000,
    seed=123,
    log_freq=500,
    l2=1.0,
    lr=0.001,
    class_balance=class_balance,
)

# %%
from snorkel.analysis.metrics import metric_score
from snorkel.analysis.utils import probs_to_preds, preds_to_probs

Y_probs_dev = label_model.predict_proba(dev_L)
Y_preds_dev = probs_to_preds(Y_probs_dev)
print(
    f"Label Model Accuracy: {metric_score(dev_labels, Y_preds_dev, probs=None, metric='accuracy')}"
)

train_proba = label_model.predict_proba(train_L)

# %% [markdown]
# ## Use Soft Labels to Train End Model

# %% [markdown]
# For simplicity and speed, we use a simple "bag of n-grams" feature representation: each data point is represented by a one-hot vector marking which words or 2-word combinations are present in the comment text.

# %% [markdown]
# ### Featurization

# %%
from sklearn.feature_extraction.text import CountVectorizer

train_tokens = [row.tweet_text for _, row in train_df.iterrows()]
dev_tokens = [row.tweet_text for _, row in dev_df.iterrows()]
val_tokens = [row.tweet_text for _, row in val_df.iterrows()]
test_tokens = [row.tweet_text for _, row in test_df.iterrows()]

vectorizer = CountVectorizer(ngram_range=(1, 2))
train_X = vectorizer.fit_transform(train_tokens).toarray().astype("float")
dev_X = vectorizer.transform(dev_tokens).toarray().astype("float")
val_X = vectorizer.transform(val_tokens).toarray().astype("float")
test_X = vectorizer.transform(test_tokens).toarray().astype("float")

# %% [markdown]
# ### Model on soft labels
# Now, we train a simple logistic regression model on the bag of words features, using labels obtained from our label model.

# %%
import tensorflow as tf


def get_lr_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))
    model.compile("Adagrad", "categorical_crossentropy")
    return model


lr_model = get_lr_model()
lr_model.fit(train_X, train_proba, epochs=200, verbose=0)
probs = lr_model.predict(test_X)
preds = probs_to_preds(probs)
print(
    f"Test Accuracy when trained with soft training labels: {metric_score(test_labels, preds=preds, metric='accuracy')}"
)

# %% [markdown]
# ### Comparison to baseline
# Finally, we can compare this to a model trained only on the gold labels in the dev set. This shows us the benefit of u

# %%
dev_model = get_lr_model()
dev_model.fit(dev_X, preds_to_probs(dev_labels, num_classes=5), epochs=200, verbose=0)
probs = dev_model.predict(test_X)
preds = probs_to_preds(probs)
print(
    f"Test Accuracy when trained with dev labels: {metric_score(test_labels, preds=preds, metric='accuracy')}"
)
