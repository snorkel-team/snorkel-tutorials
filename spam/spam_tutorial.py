# %% [markdown]
# # Introductory Snorkel Tutorial: Spam Detection

# %% [markdown]
# In this tutorial, we will walk through the process of using `Snorkel` to classify YouTube comments as `SPAM` or `HAM` (not spam). For an overview of Snorkel, visit [snorkel.stanford.edu](http://snorkel.stanford.edu).
#
# For our task, we have access to a large amount of *unlabeled data*, which can be prohibitively expensive and slow to label manually. We therefore turn to weak supervision using *labeling functions*, or noisy, programmatic heuristics, to assign labels to unlabeled training data efficiently. We also have access to a small amount of labeled data, which we only use for evaluation purposes.
#
# The tutorial is divided into four parts:
# 1. **Loading Data**: We load a [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) from Kaggle.
#
# 2. **Writing Labeling Functions**: We write Python programs that take as input a data point and assign labels (or abstain) using heuristics, pattern matching, and third-party models.
#
# 3. **Combining Labels with the Label Model**: We use the outputs of the labeling functions over the training set as input to the label model, which assings probabilistic labels to the training set.
#
# 4. **Training a Classifier**: We train a classifier that can predict labels for *any* YouTube comment (not just the ones labeled by the labeling functions) using the probabilistic training labels from step 3.

# %% [markdown]
# ### Task: Spam Detection

# %% [markdown]
# We use a [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) that consists of YouTube comments from 5 videos. The task is to classify each comment as being `SPAM`, irrelevant or inappropriate messages, or `HAM`, comments relevant to the video.
#
# For example, the following comments are `SPAM`:
#
#         Subscribe to me for free Android games, apps..
#
#         Please check out my vidios
#
#         Subscribe to me and I'll subscribe back!!!
#
# and these are `HAM`:
#
#         3:46 so cute!
#
#         This looks so fun and it's a good song
#
#         This is a weird video.

# %% [markdown]
# ### Data Splits in Snorkel
#
# We split our data into 4 sets:
# * **training set**: This is the largest split of the dataset with no labels
# * **Validation (Valid) Set**: A labeled set used to tune hyperparameters for the end classifier.
# * **Test Set**: A labeled set used to evaluate the classifier. Note that users should not be looking at the test set and use it only for the final evaluation step.
#
# While it is possible to write labeling functions without any labeled data and evaluate only with the classifier, it is often useful to have some labeled data to estimate how the labeling functions are doing. For example, we can calculate the accuracy and coverage of the labeling functions over this labeled set, which can help guide how to edit existing labeling functions and/or what type of labeling functions to add.
#
# We refer to this as the **development (dev) set**. This can either be a small labeled subset of the training set (e.g. 100 data points) or we can use the valid set as the dev set. Note that in the latter case, the valid set will not be representative of the test set since the labeling functions are created to fit specifically to the validation set.

# %% [markdown]
# ## 1. Load data

# %% [markdown]
# We load the Kaggle dataset and create Pandas dataframe objects for each of the sets described above. Each dataframe consists of the following fields:
# * **COMMENT_ID**: Comment ID
# * **AUTHOR_ID**: Author ID
# * **DATE**: Date and time the comment was posted
# * **CONTENT**: The raw text content of the comment
# * **LABEL**: Whether the comment is `SPAM` (1) or `HAM` (2)
# * **VIDEO_ID**: The video the comment is associated with
#
# We start by loading our data.
# The `load_spam_dataset()` method downloads the raw csv files from the internet, divides them into splits, converts them into dataframes, and shuffles them.
# As mentioned above, the dataset contains comments from 5 of the most popular YouTube videos during a particular timeframe in 2014 and 2015.
# * The first four videos' comments are combined to form the `train` set. This set has no gold labels.
# * The `dev` set is a random sample of 200 `DataPoints` from the `train` set with gold labels added.
# * The fifth video is split 50/50 between a validation set (`valid`) and `test` set.

# %%
from spam.utils import load_spam_dataset

df_train, df_dev, df_valid, df_test = load_spam_dataset()


# %% [markdown]
# The class distribution varies slightly from class to class, but all are approximately class-balanced.

# %%
from collections import Counter

# For clarity, we'll define constants to represent the class labels for spam, ham, and abstaining.
ABSTAIN = 0
SPAM = 1
HAM = 2

for split_name, df in [
    ("train", df_train),
    ("dev", df_dev),
    ("valid", df_valid),
    ("test", df_test),
]:
    counts = Counter(df["LABEL"].values)
    print(
        f"{split_name.upper():<6} {counts[SPAM] * 100/sum(counts.values()):0.1f}% SPAM"
    )

# %% [markdown]
# Taking a peek at our data, we see that for each `DataPoint`, we have the following fields:
# * `COMMENT_ID`: A unique identifier
# * `AUTHOR`: The user who made the comment
# * `DATE`: The date the comment was made
# * `CONTENT`: The comment text
# * `LABEL`:
#     * 0 = UNKNOWN/ABSTAIN
#     * 1 = SPAM
#     * 2 = HAM (not spam)
# * `VIDEO_ID`: Which of the five videos in the dataset the comment came from

# %%
# Don't truncate text fields in the display
import pandas as pd

pd.set_option("display.max_colwidth", 0)

df_dev.sample(5, random_state=123)

# %% [markdown]
# ## 2. Write Labeling Functions (LFs)

# %% [markdown]
# * What's an LF
#     * Why are they awesome
# * Can be many types:
#     * keyword
#     * pattern-match
#     * heuristic
#     * third-party models
#     * distant supervision
#     * crowdworkers (non-expert)
# * Typically an iterative process
#     * Look at examples for ideas
#     * Write an LF
#     * Check performance on dev set
#     * Balance accuracy/coverage

# %% [markdown]
# ### a) Look at examples for ideas

# %% [markdown]
# * Look at 10 examples; got any ideas?

# %%
# Display just the text and label
df_dev[["CONTENT", "LABEL"]].sample(10, random_state=123)

# %%
# for i, x in df_dev.iterrows():
#     if "please" in x.CONTENT:
#         print(x.CONTENT)

# %% [markdown]
# ### b) Write an LF

# %% [markdown]
# The simplest way to create labeling functions in Snorkel is with the `@labeling_function()` decorator, which wraps a function for evaluating on a single `DataPoint` (in this case, a row of the dataframe).
#
# Looking at samples of our data, we see multiple messages where spammers are trying to get viewers to look at "my channel" or "my video," so we write a simple LF that labels an example as spam if it includes the word "my".

# %%
from snorkel.labeling.lf import labeling_function

# We initialize an empty list that we'll add our LFs to as we create them
lfs = []


@labeling_function()
def keyword_my(x):
    """Many spam comments talk about 'my channel', 'my video', etc."""
    return SPAM if "my" in x.CONTENT.lower() else ABSTAIN


lfs.append(keyword_my)

# %% [markdown]
# To apply one or more LFs that we've written to a collection of `DataPoints`, we use an `LFApplier`.
#
# Because our `DataPoints` are represented with a Pandas dataframe in this tutorial, we use the `PandasLFApplier` class.

# %%
from snorkel.labeling.apply import PandasLFApplier

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)

# %% [markdown]
# The output of the `apply()` method is a sparse label matrix which we generally refer to as `L`.

# %%
L_train

# %% [markdown]
# ### c) Check performance on dev set

# %% [markdown]
# We can easily calculate the coverage of this LF by hand (i.e., the percentage of the dataset that it labels) as follows:

# %%
coverage = L_train.nnz / L_train.shape[0]
print(f"Coverage: {coverage}")

# %% [markdown]
# To get an estimate of its accuracy, we can label the development set with it and compare that to the few gold labels we do have.
#
# Note that we don't want to penalize the LF for examples where it abstained, so we filter out both the predictions and the gold labels where the prediction is `ABSTAIN`.

# %%
import numpy as np

L_dev = applier.apply(df_dev)
L_dev_array = np.asarray(L_dev.todense()).squeeze()

Y_dev = df_dev["LABEL"].values

accuracy = ((L_dev_array == Y_dev)[L_dev_array != ABSTAIN]).sum() / (
    L_dev_array != ABSTAIN
).sum()
print(f"Accuracy: {accuracy}")

# %% [markdown]
# Alternatively, you can use the provided `metric_score()` helper method, which allows you to specify a metric to calculate and certain classes to ignore (such as ABSTAIN).

# %%
from snorkel.analysis.metrics import metric_score

# Calculate accuracy, ignore all examples for which the predicted label is ABSTAIN
# TODO: drop probs=None
accuracy = metric_score(
    golds=Y_dev,
    preds=L_dev_array,
    probs=None,
    metric="accuracy",
    filter_dict={"preds": [ABSTAIN]},
)
print(f"Accuracy: {accuracy}")

# %% [markdown]
# You can also use the helper method `lf_summary()` to report the following summary statistics for multiple LFs at once:
# * Polarity: The set of labels this LF outputs
# * Coverage: The fraction of the dataset the LF labels
# * Overlaps: The fraction of the dataset where this LF and at least one other LF label
# * Conflicts: The fraction of the dataset where this LF and at least one other LF label and disagree
# * Correct: The number of `DataPoints` this LF labels correctly (if gold labels are provided)
# * Incorrect: The number of `DataPoints` this LF labels incorrectly (if gold labels are provided)
# * Emp. Acc.: The empirical accuracy of this LF (if gold labels are provided)

# %%
from snorkel.labeling.analysis import lf_summary

lf_names = [lf.name for lf in lfs]
lf_summary(L=L_dev, Y=Y_dev, lf_names=lf_names)

# %% [markdown]
# ### d) Balance accuracy/coverage

# %% [markdown]
# Often, by looking at the examples that an LF does and doesn't label, we can get ideas for how to improve it.
#
# The helper method `error_buckets()` groups examples by their predicted label and true label, so `buckets[(1, 2)]` will contain the indices of examples that that the LF labeled 1 (SPAM) that were actually of class 2 (HAM).

# %%
from snorkel.analysis.error_analysis import error_buckets

buckets = error_buckets(Y_dev, L_dev_array)
df_dev[["CONTENT", "LABEL"]].iloc[buckets[(1, 2)]].head()

# %% [markdown]
# On the other hand, `buckets[(1, 1)]` contains SPAM examples it labeled correctly.

# %%
df_dev[["CONTENT", "LABEL"]].iloc[buckets[(1, 1)]].head()


# %% [markdown]
# Looking at these examples, we may notice that much of the time when "my" is used, it's referring to "my channel". We can update our LF to see how making this change affects accuracy and coverage.

# %%
@labeling_function()
def keywords_my_channel(x):
    return SPAM if "my channel" in x.CONTENT.lower() else ABSTAIN


lfs = [keywords_my_channel]
applier = PandasLFApplier(lfs)
L_dev = applier.apply(df_dev)
lf_names = [lf.name for lf in lfs]
lf_summary(L=L_dev, Y=Y_dev)

# %% [markdown]
# In this case, accuracy does improve a bit, but it was already fairly accurate to begin with, and "tightening" the LF like this causes the coverage drops significantly, so we'll stick with the original LF.

# %% [markdown]
# ## More Labeling Functions

# %% [markdown]
# If a single LF had high enough coverage to label our entire test dataset accurately, then we wouldn't need a classifier at all; we could just use that single simple heuristic to complete the task. But most problems are not that simple. Instead, we usually need to **combine multiple LFs** to label our dataset, both to increase the size of the generated training set (since we can't generate training labels for data points that all LFs abstained on) and to improve the overall accuracy of the training labels we generate by factoring in multiple different signals.
#
# In the following subsections, we'll show just a few of the many types of LFs that you could write to generate a training dataset for this problem.

# %% [markdown]
# ### i. Keyword LFs

# %% [markdown]
# For text applications, some of the simplest LFs to write are often just keyword lookups.

# %%
lfs = []


@labeling_function()
def keyword_my(x):
    """Spam comments talk about 'my channel', 'my video', etc."""
    return SPAM if "my" in x.CONTENT.lower() else ABSTAIN


lfs.append(keyword_my)


@labeling_function()
def lf_subscribe(x):
    """Spam comments ask users to subscribe to their channels."""
    return SPAM if "subscribe" in x.CONTENT else 0


lfs.append(lf_subscribe)


@labeling_function()
def lf_link(x):
    """Spam comments post links to other channels."""
    return SPAM if "http" in x.CONTENT.lower() else 0


lfs.append(lf_link)


@labeling_function()
def lf_please(x):
    """Spam comments make requests rather than commenting."""
    return (
        SPAM
        if any([word in x.CONTENT.lower() for word in ["please", "plz"]])
        else ABSTAIN
    )


lfs.append(lf_please)


@labeling_function()
def lf_song(x):
    """Ham comments actually talk about the video's content."""
    return HAM if "song" in x.CONTENT.lower() else ABSTAIN


lfs.append(lf_song)

# %% [markdown]
# ### ii. Pattern-matching LFs (Regular Expressions)

# %% [markdown]
# If we want a little more control over a keyword search, we can look for regular expressions instead.

# %%
import re


@labeling_function()
def regex_check_out(x):
    """Spam comments say 'check out my video', 'check it out', etc."""
    return SPAM if re.search(r"check.*out", x.CONTENT, flags=re.I) else ABSTAIN


lfs.append(regex_check_out)


# %% [markdown]
# ### iii.  Heuristic LFs

# %% [markdown]
# There may other heuristics or "rules of thumb" that you come up with as you look at the data.
# So long as you can express it in a function, it's a viable LF!

# %%
@labeling_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return HAM if len(x.CONTENT.split()) < 5 else ABSTAIN


lfs.append(short_comment)


# @labeling_function()
# def short_word_lengths(x):
#     """Ham comments tend to have shorter words."""
#     words = x.CONTENT.split()
#     lengths = [len(word) for word in words]
#     mean_word_length = sum(lengths) / len(lengths)
#     return HAM if mean_word_length < 4 else ABSTAIN
# lfs.append(short_word_lengths)

# %% [markdown]
# ### Adding Preprocessors

# %% [markdown]
# Some LFs rely on fields that aren't present in the raw data, but can be derived from it. We can enrich our data (providing more fields for the LFs to refer to) using `Preprocessors`.
#
# For example, we can use the fantastic NLP tool [spaCy](https://spacy.io/) to add lemmas, part-of-speech (pos) tags, etc. to each token.

# %%
# Download the spacy english model
# ! python -m spacy download en_core_web_sm

# %%
from snorkel.labeling.preprocess.nlp import SpacyPreprocessor

# The SpacyPreprocessor parses the text in text_field and
# stores the new enriched representation in doc_field
spacy = SpacyPreprocessor(text_field="CONTENT", doc_field="doc", memoize=True)


@labeling_function(preprocessors=[spacy])
def has_person(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN


lfs.append(has_person)

# %% [markdown]
# ### iv. Third-party Model LFs

# %% [markdown]
# We can also utilize other models, including ones trained for other tasks that are related to, but not the same as, the one we care about.
#
# For example, the [TextBlob](https://textblob.readthedocs.io/en/dev/index.html) tool provides a pretrained sentiment analyzer. Our spam classification task is not the same as sentiment classification, but it turns out that SPAM and HAM comments have different distributions of sentiment scores, with HAM having more positive/subjective sentiments.

# %%
import matplotlib.pyplot as plt
from textblob import TextBlob

spam_polarities = [
    TextBlob(x.CONTENT).sentiment.polarity
    for i, x in df_dev.iterrows()
    if x.LABEL == SPAM
]
ham_polarities = [
    TextBlob(x.CONTENT).sentiment.polarity
    for i, x in df_dev.iterrows()
    if x.LABEL == HAM
]

_ = plt.hist([spam_polarities, ham_polarities])

# %%
from textblob import TextBlob


@labeling_function()
def textblob_polarity(x):
    return 2 if TextBlob(x.CONTENT).sentiment.polarity > 0.3 else 0


lfs.append(textblob_polarity)


@labeling_function()
def textblob_subjectivity(x):
    return 2 if TextBlob(x.CONTENT).sentiment.subjectivity > 0.9 else 0


lfs.append(textblob_subjectivity)

# %% [markdown]
# ### v. Write your own LFs

# %% [markdown]
# This tutorial demonstrates just a handful of the types of LFs that one might write for this task.
# Many of these are no doubt suboptimal.
# The strength of this approach, however, is that the LF abstraction provides a flexible interface for conveying a huge variety of supervision signals, and the `LabelModel` is able to denoise these signals, reducing the need for painstaking manual fine-tuning.
#
# You can uncomment the cell below to write one or more of your own LFs.
# Don't forget to add them to the list of `lfs` so that they are included by the `LFApplier` in the next section.

# %%
# @labeling_function()
# def my_lf(x):
#     pass
# lfs.append(my_lf)

# %% [markdown]
# ### Apply LFs

# %% [markdown]
# With our full set of LFs (including any you wrote), we can now apply these once again with `LFApplier` to get our the label matrices for the `train` and `dev` splits. We'll use the `train` split's label matrix to generate training labels with the Label Model. The `dev` split's label model is primarily helpful for looking at summary statistics.
#
# Note that the `pandas` format provides an easy interface that many practioners are familiar with, but it is also less optimized for scale. For larger datasets, more compute-intensive LFs, or larger LF sets, you may decide to use one of the other supported data formats such as `dask` or `spark` dataframes, and their corresponding applier objects.

# %%
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_dev = applier.apply(df_dev)

lf_names = [lf.name for lf in lfs]
lf_summary(L=L_dev, Y=Y_dev, lf_names=lf_names)

# %% [markdown]
# We see that our labeling functions vary in coverage, accuracy, and how much they overlap/conflict with one another.
# We can view a histogram of how many weak labels the `DataPoints` in our dev set have to get an idea of our total coverage.

# %%
# TODO: Move plot_label_frequency() to core snorkel repo
import matplotlib.pyplot as plt


def plot_label_frequency(L):
    plt.hist(np.asarray((L != 0).sum(axis=1)), density=True, bins=range(L.shape[1]))


plot_label_frequency(L_train)

# %% [markdown]
# We see that over half of our training dataset `DataPoints` have 0 or 1 weak labels.
# Fortunately, the signal we do have can be used to train a classifier with a larger feature set than just these labeling functions that we've created, allowing it to generalize beyond what we've specified.

# %% [markdown]
# ## 3. Combine with Label Model

# %% [markdown]
# Our goal is now to convert these many weak labels into a single _noise-aware_ probabilistic (or confidence-weighted) label per `DataPoint`.
# A simple baseline for doing this is to take the majority vote on a per-`DataPoint` basis: if more LFs voted SPAM than HAM, label it SPAM (and vice versa).

# %%
from snorkel.labeling.model import MajorityLabelVoter

mv_model = MajorityLabelVoter()
mv_model.score(L_dev, Y_dev)

# %% [markdown]
# However, as we can clearly see by looking the summary statistics of our LFs, they are not all equally accurate, and should ideally not be treated identically. In addition to having varied accuracies and coverages, LFs may be correlated, resulting in certain signals being overrepresented in a majority-vote-based model. To handle these issues appropriately, we will instead use a more sophisticated Snorkel `LabelModel` to combine our weak labels.
#
# This model will ultimately produce a single set of noise-aware training labels, which are probabilistic or confidence-weighted labels. We will then use these labels to train a classifier for our task. For more technical details of this overall approach, see our [NeurIPS 2016](https://arxiv.org/abs/1605.07723) and [AAAI 2019](https://arxiv.org/abs/1810.02840) papers.
#
# Note that no gold labels are used during the training process; the `LabelModel` is able to estimate the accuracies of the labeling functions using only the weak label matrix as input. (See the [TODO: dependency learning](TBD) tutorial for a demonstration of how to learn correlations as well).

# %%
from snorkel.labeling.model import LabelModel

# TODO: get more frequent logging statement printing
label_model = LabelModel(cardinality=2, verbose=True, seed=123)
label_model.train_model(L_train, n_epochs=300, log_train_every=25)


# %% [markdown]
# We can confirm that our resulting predicted labels are probabilistic, with the points we are least certain about having labels close to 0.5. The following histogram shows the confidences we have that each `DataPoint` has the label SPAM.

# %%
def plot_probabilities_histogram(Y_probs):
    plt.hist(Y_probs[:, 0])


Y_probs_train = label_model.predict_proba(L_train)
plot_probabilities_histogram(Y_probs_train)

# %%
label_model.score(L_dev, Y_dev)

# %% [markdown]
# While our `LabelModel` does improve over the majority vote baseline, it is still somewhat limited as a classifier.
# For example, many of our `DataPoints` have few or no LFs voting on them.
# We will now train a discriminative classifier with this training set to see if we can improve performance further.

# %% [markdown]
# ## 4. Predict with Classifier

# %% [markdown]
# * Now train classifier
#     * Can use any third-party classifier (plug into your existing pipelines!)
#     * Some libraries natively support probabilistic labels (us, TF); for others, can round.

# %% [markdown]
# * Convert label convention

# %%
from snorkel.analysis.utils import probs_to_preds, convert_labels

# Y_train = df_train['LABEL'].map({1: 1, 2: 0}) # This will not be available
Y_train = Y_probs_train
Y_dev = convert_labels(df_dev["LABEL"].values, "categorical", "onezero")
Y_valid = convert_labels(df_valid["LABEL"].values, "categorical", "onezero")
Y_test = convert_labels(df_test["LABEL"].values, "categorical", "onezero")

# %% [markdown]
# * Use bag-of-ngrams as features

# %%
from sklearn.feature_extraction.text import CountVectorizer

words_train = [row.CONTENT for i, row in df_train.iterrows()]
words_dev = [row.CONTENT for i, row in df_dev.iterrows()]
words_valid = [row.CONTENT for i, row in df_valid.iterrows()]
words_test = [row.CONTENT for i, row in df_test.iterrows()]

vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(words_train)
X_dev = vectorizer.transform(words_dev)
X_valid = vectorizer.transform(words_valid)
X_test = vectorizer.transform(words_test)

# %% [markdown]
# * Filter out examples with no labels

# %%
mask = np.asarray(L_train.sum(axis=1) > 0).squeeze()
X_train = X_train[mask, :]
Y_train = Y_train[mask]

# %% [markdown]
# ### Keras Classifier

# %% [markdown]
# * TBD

# %%
# TODO:
# import simple logistic regression classifier in Keras
# train on noise-aware loss
# score
# bonus: peek at performance w/r/t using hard (int) labels i/o soft (float) labels

# %% [markdown]
# ### Scikit-learn Classifier

# %% [markdown]
# * TBD
# * Rounding to hard labels

# %%
Y_train_rounded = convert_labels(probs_to_preds(Y_train), "categorical", "onezero")

# %%
from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression()
sklearn_model.fit(X_train, Y_train_rounded)

sklearn_model.score(X_valid, Y_valid)

# %% [markdown]
# * Compare with training on dev directly (see, we did better)
#     * And we could do even better with more raw unlabeled data

# %%
X_dev = vectorizer.fit_transform(words_dev)
X_valid = vectorizer.transform(words_valid)
X_test = vectorizer.transform(words_test)

sklearn_model.fit(X_dev, Y_dev)
sklearn_model.score(X_valid, Y_valid)

# %% [markdown]
# ### Evaluate on Test Set

# %% [markdown]
# * With all ablations done, now evaluate on test

# %%
# TBD
