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
# conflicting sentiment labels to the same tweet. Our goal is to label each tweet as either
# positive or negative.
#
# This is a common issue when dealing with crowdsourced labeling workloads.
# We've also altered the data set to reflect a realistic crowdsourcing pipeline
# where only a subset of our full training set have recieved crowd labels.
# We'll encode the crowd labels themselves as labeling functions in order to learn trust
# weights for each crowdworker, and write a few heuristic labeling functions to cover the
# data points without crowd labels.
# Snorkel's ability to build high-quality datasets from multiple noisy labeling
# signals makes it an ideal framework to approach this problem.

# %% [markdown]
# We start by loading our data. It has 632 examples. We take 50 for our development set and 50 for our test set. The remaining 187 examples form our training set. 100 of the examples have crowd labels, and the remaining 87 do not. This data set is very small, and we're primarily using it for demonstration purposes.
#
# The labels above have been mapped to integers, which we show here.

# %% [markdown]
# ## Loading Crowdsourcing Dataset

# %%
import os

if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("crowdsourcing")

# %%
from data import load_data, answer_mapping

crowd_answers, df_train, df_dev, df_test = load_data()
Y_dev = df_dev.sentiment.values
Y_test = df_test.sentiment.values

print("Answer to int mapping:")
for k, v in sorted(answer_mapping.items(), key=lambda kv: kv[1]):
    print(f"{k:<50}{v}")

# %% [markdown]
# First, let's take a look at our development set to get a sense of what the tweets look like.

# %%
df_dev.head()

# %% [markdown]
# Now let's take a look at the crowd labels. We'll convert these into labeling functions.

# %%
crowd_answers.head()

# %% [markdown]
# ## Writing Labeling Functions
# Each crowd worker can be thought of as a single labeling function, as each worker labels a subset of examples, and may have errors or conflicting answers with other workers / labeling functions. So we create one labeling function per worker. We'll simply return the label the worker submitted for a given tweet, and abstain if they didn't submit an answer for it.

# %% [markdown]
# ### Crowd worker labeling functions

# %%
labels_by_annotator = crowd_answers.groupby("worker_id")
worker_dicts = {}
for worker_id in labels_by_annotator.groups:
    worker_df = labels_by_annotator.get_group(worker_id)[["answer"]]
    v = set(worker_df.answer.tolist())
    if len(worker_df) > 10:
        worker_dicts[worker_id] = dict(zip(worker_df.index, worker_df.answer))

print("Number of workers:", len(worker_dicts))

# %%
from snorkel.labeling.lf import LabelingFunction


def f_pos(x, worker_dict):
    label = worker_dict.get(x.tweet_id)
    return 1 if label == 1 else -1


def f_neg(x, worker_dict):
    label = worker_dict.get(x.tweet_id)
    return 0 if label == 0 else -1


def get_worker_labeling_function(worker_id, f):
    worker_dict = worker_dicts[worker_id]
    name = f"worker_{worker_id}"
    return LabelingFunction(name, f=f, resources={"worker_dict": worker_dict})


worker_lfs_pos = [
    get_worker_labeling_function(worker_id, f_pos) for worker_id in worker_dicts
]
worker_lfs_neg = [
    get_worker_labeling_function(worker_id, f_neg) for worker_id in worker_dicts
]

# %% [markdown]
# Let's take a quick look at how well they do on the development set.

# %%
from snorkel.labeling.apply import PandasLFApplier

lfs = worker_lfs_pos + worker_lfs_neg
lf_names = [lf.name for lf in lfs]

applier = PandasLFApplier(lfs)
L_dev = applier.apply(df_dev)

# %%
from snorkel.labeling.analysis import LFAnalysis

LFAnalysis(L_dev).lf_summary(Y_dev, lf_names=lf_names).head(10)

# %% [markdown]
# ### Additional labeling functions
#
# We can mix the crowd worker labeling functions with labeling functions of other types.
# We'll use a few varied approaches and use the label model learn how to combine their values.

# %%
from snorkel.labeling.lf import labeling_function
from snorkel.labeling.preprocess import preprocessor
from textblob import TextBlob


@preprocessor()
def textblob_polarity(x):
    scores = TextBlob(x.tweet_text)
    x.polarity = scores.polarity
    return x


textblob_polarity.memoize = True


@labeling_function(preprocessors=[textblob_polarity])
def polarity_positive(x):
    return 1 if x.polarity > 0.3 else -1


@labeling_function(preprocessors=[textblob_polarity])
def polarity_negative(x):
    return 0 if x.polarity < -0.25 else -1


@labeling_function(preprocessors=[textblob_polarity])
def polarity_negative_2(x):
    return 0 if x.polarity <= 0.3 else -1


# %% [markdown]
# ### Applying labeling functions to the training set

# %%
from snorkel.labeling.apply import PandasLFApplier

text_lfs = [polarity_positive, polarity_negative, polarity_negative_2]
lfs = text_lfs + worker_lfs_pos + worker_lfs_neg
lf_names = [lf.name for lf in lfs]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_dev = applier.apply(df_dev)

# %%
LFAnalysis(L_dev).lf_summary(Y_dev, lf_names=lf_names).head(10)

# %% [markdown]
# ## Train Label Model And Generate Soft Labels

# %%
from snorkel.labeling.model.label_model import LabelModel

# Train label model.
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=100, seed=123, log_freq=20, l2=0.1, lr=0.01)

# %% [markdown]
# As a spot-check for the quality of our label model, we'll score it on the dev set.

# %%
from snorkel.analysis.metrics import metric_score
from snorkel.analysis.utils import probs_to_preds

Y_dev_prob = label_model.predict_proba(L_dev)
Y_dev_pred = probs_to_preds(Y_dev_prob)

acc = metric_score(Y_dev, Y_dev_pred, probs=None, metric="accuracy")
print(f"Label Model Accuracy: {acc:.3f}")

# %% [markdown]
# Look at that, we get perfect accuracy on the development set. This is due to the abundance of high quality crowd worker labels. In order to train a discriminative model, let's generate a set of probabilistic labels for the training set.

# %%
Y_train_prob = label_model.predict_proba(L_train)

# %% [markdown]
# ## Use Soft Labels to Train End Model

# %% [markdown]
# For simplicity and speed, we use a simple "bag of n-grams" feature representation: each data point is represented by a one-hot vector marking which words or 2-word combinations are present in the comment text.

# %% [markdown]
# ### Featurization

# %%
from sklearn.feature_extraction.text import CountVectorizer

train_tokens = [row.tweet_text for _, row in df_train.iterrows()]
test_tokens = [row.tweet_text for _, row in df_test.iterrows()]

vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_tokens).toarray().astype("float")
X_test = vectorizer.transform(test_tokens).toarray().astype("float")

# %% [markdown]
# ### Model on soft labels
# Now, we train a simple MLP model on the bag-of-words features, using labels obtained from our label model.

# %%
import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
model.compile("Adam", "categorical_crossentropy")
callbacks = model.fit(X_train, Y_train_prob, epochs=100, verbose=0)

# %%
probs = model.predict(X_test)
preds = probs_to_preds(probs)
acc = metric_score(Y_test, preds=preds, metric="accuracy")
print(f"Test Accuracy when trained with soft training labels: {acc:.3f}")
