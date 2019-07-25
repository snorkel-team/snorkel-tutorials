# %% [markdown]
# # Introductory Snorkel Tutorial: Spam Detection

# %% [markdown]
# In this tutorial, we will walk through the process of using `Snorkel` to classify YouTube comments as `SPAM` or `HAM` (not spam). For an overview of Snorkel, visit [snorkel.org](http://snorkel.org).
#
# For our task, we have access to a large amount of *unlabeled data*, which can be prohibitively expensive and slow to label manually. We therefore turn to weak supervision using *labeling functions*, or noisy, programmatic heuristics, to assign labels to unlabeled training data efficiently. We also have access to a small amount of labeled data, which we only use for evaluation purposes.
#
# The tutorial is divided into four parts:
# 1. **Loading Data**: We load a [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) from Kaggle.
#
# 2. **Writing Labeling Functions**: We write Python programs that take as input a data point and assign labels (or abstain) using heuristics, pattern matching, and third-party models.
#
# 3. **Combining Weak Labels with the Label Model**: We use the outputs of the labeling functions over the training set as input to the label model, which assings probabilistic labels to the training set.
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
# * **Training Set**: The largest split of the dataset. We do not have ground truth or "gold" labels for these data points; we will be generating their labels with weak supervision.
# * **Development Set**: \[Optional\] A small labeled subset of the training data (e.g. 100 points) to guide LF iteration. See note below.
# * **Validation Set**: A labeled set used to tune hyperparameters and/or perform early stopping while training the classifier.
# * **Test Set**: A labeled set for final evaluation of our classifier. This set should only be used for final evaluation, _not_ error analysis.
#
#
# While it is possible to develop labeling functions on the unlabeled training set only, users often find it more time-efficient to label a small dev set to provide a quick approximate signal on the accuracies and failure modes of their LFs (rather than scrolling through training examples and mentally assessing approximate accuracy).
# Alternatively, users sometimes will have the validation set also serve as the development set.
# Do the latter only with caution: because the labeling functions will be based on examples from the valid set, the valid set will no longer be an unbiased proxy for the test set.

# %% [markdown]
# ## 1. Loading Data

# %% [markdown]
# We load the Kaggle dataset and create Pandas dataframe objects for each of the sets described above. Each dataframe consists of the following fields:
# * **author**: Username of the comment author
# * **data**: Date and time the comment was posted
# * **text**: Raw text content of the comment
# * **label**: Whether the comment is `SPAM` (1), `HAM` (0), or `UNKNOWN/ABSTAIN` (-1)
# * **video**: Video the comment is associated with
#
# We start by loading our data.
# The `load_spam_dataset()` method downloads the raw csv files from the internet, divides them into splits, converts them into dataframes, and shuffles them.
# As mentioned above, the dataset contains comments from 5 of the most popular YouTube videos during a particular timeframe in 2014 and 2015.
# * The first four videos' comments are combined to form the `train` set. This set has no gold labels.
# * The `dev` set is a random sample of 200 data points from the `train` set with gold labels added.
# * The fifth video is split 50/50 between a validation set (`valid`) and `test` set.

# %%
import os

if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("spam")

# %%
from utils import load_spam_dataset

df_train, df_dev, df_valid, df_test = load_spam_dataset()

# We pull out the label vectors for ease of use later
Y_dev = df_dev["label"].values
Y_valid = df_valid["label"].values
Y_test = df_test["label"].values


# %% [markdown]
# Let's view a few examples

# %%
import pandas as pd

# Don't truncate text fields in the display
pd.set_option("display.max_colwidth", 0)

df_dev.sample(5, random_state=123)

# %% [markdown]
# The class distribution varies slightly from class to class, but all are approximately class-balanced.

# %%
from collections import Counter

# For clarity, we define constants to represent the class labels for spam, ham, and abstaining.
ABSTAIN = -1
HAM = 0
SPAM = 1

for split_name, df in [("dev", df_dev), ("valid", df_valid), ("test", df_test)]:
    counts = Counter(df["label"].values)
    num_points = sum(counts.values())
    print(f"{split_name.upper():<6} {counts[SPAM] * 100 / num_points:0.1f}% SPAM")

# %% [markdown]
# ## 2. Write Labeling Functions (LFs)

# %% [markdown]
# **Labeling functions (LFs) help users encode domain knowledge and other supervision sources programmatically.**
#
# LFs are heuristics that take as input a data point and either assign a label to it (in this case, `HAM` or `SPAM`) or abstain (don't assign any label). Labeling functions can be *noisy*: they don't have perfect accuracy and don't have to label every data point.
#
# Because their only requirement is that they map a data point a label (or abstain), they can wrap a wide variety of forms of supervision. Examples include, but are not limited to:
# * Keyword searches: looking for specific words in a sentence
# * Pattern matching: looking for specific syntactical patterns
# * Third-party models: using an pre-trained model (usually a model for a different task than the one at hand)
# * Distant supervision: using external knowledge base
# * Crowdworker labels: treating each crowdworker as a black-box function that assigns labels to subsets of the data
#
# The process of **developing LFs** is iterative and usually consists of:
# * Writing a function
# * Estimating its performance by looking at labeled examples in the training set or dev set
# * Iterating to improve coverage or accuracy as necessary.
# Balancing accuracy and coverage for specific labeling functions as well as the overall set of LFs developed is often a trade-off, and depending on the use case, users may tend to prefer one over the other.
#
# Once multiple LFs have been created, users can look at data points that have received no labels so far (or many conflicting labels) to get ideas for new LFs to write.
# Another thing to check is whether there are any classes for which there are very few LFs, so that additional LFs can be written to cover that part of the dataset.
# **Note, however, that it is not necessary for LFs to assign labels to every data point;** in fact, most of the time your LFs will not have perfect dataset-wide coverage.
# We rely on the fact that the classifier that trains on labels from Snorkel has the power to _generalize_ and can therefore learn a good representation of the data even if the each data point in the training set does not receive a label from any LFs.
#
#

# %% [markdown]
# ### a) Look at examples for ideas

# %% [markdown]
# We begin the process of writing LFs by looking at some examples in the dev set.

# %%
# Display just the text and label
df_dev[["text", "label"]].sample(10, random_state=123)

# %% [markdown]
# ### b) Write an LF

# %% [markdown]
# The recommended way to create labeling functions in Snorkel is with the `@labeling_function()` decorator, which wraps a function for evaluating on a single data point (in this case, a row of the dataframe).
#
# Looking at samples of our data, we see multiple messages where spammers are trying to get viewers to look at "my channel" or "my video," so we write a simple LF that labels an example as `SPAM` if it includes the word "my" and otherwise abstains.

# %%
from snorkel.labeling.lf import labeling_function


@labeling_function()
def keyword_my(x):
    """Many spam comments talk about 'my channel', 'my video', etc."""
    return SPAM if "my" in x.text.lower() else ABSTAIN


lfs = [keyword_my]

# %% [markdown]
# To apply one or more LFs that we've written to a collection of data points, we use an `LFApplier`.
#
# Because our data points are represented with a Pandas dataframe in this tutorial, we use the `PandasLFApplier` class.

# %%
from snorkel.labeling.apply import PandasLFApplier

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)

# %% [markdown]
# The output of the `apply()` method is a label matrix which we generally refer to as `L` (or `L_[split name]`).

# %%
L_train

# %% [markdown]
# ### c) Check performance on dev set

# %% [markdown]
# We can easily calculate the coverage of this LF by hand (i.e., the percentage of the dataset that it labels) as follows:

# %%
import numpy as np

coverage = np.sum(L_train != ABSTAIN) / L_train.shape[0]
print(f"Coverage: {coverage:.3f}")

# %% [markdown]
# To get an estimate of its accuracy, we can label the development set with it and compare that to the few gold labels we do have.
#
# Note that we don't want to penalize the LF for examples where it abstained, so we calculate the accuracy only over those examples where the LF output a label.

# %%
import numpy as np

L_dev = applier.apply(df_dev)
L_dev_array = L_dev.squeeze()

correct = L_dev_array == Y_dev
labeled = L_dev_array != ABSTAIN
accuracy = (correct * labeled).sum() / labeled.sum()
print(f"Accuracy: {accuracy:.3f}")

# %% [markdown]
# Alternatively, you can use the provided `metric_score()` helper method, which allows you to specify a metric to calculate and certain classes to ignore (such as ABSTAIN).

# %%
from snorkel.analysis.metrics import metric_score

# Calculate accuracy, ignore all examples for which the predicted label is ABSTAIN
accuracy = metric_score(
    golds=Y_dev, preds=L_dev_array, metric="accuracy", filter_dict={"preds": [ABSTAIN]}
)
print(f"Accuracy: {accuracy:.3f}")

# %% [markdown]
# You can also use the helper class `LFAnalysis()` to report the following summary statistics for multiple LFs at once:
# * Polarity: The set of labels this LF outputs
# * Coverage: The fraction of the dataset the LF labels
# * Overlaps: The fraction of the dataset where this LF and at least one other LF label
# * Conflicts: The fraction of the dataset where this LF and at least one other LF label and disagree
# * Correct: The number of data points this LF labels correctly (if gold labels are provided)
# * Incorrect: The number of data points this LF labels incorrectly (if gold labels are provided)
# * Emp. Acc.: The empirical accuracy of this LF (if gold labels are provided)
#
# The overlaps percentage gives us a rough sense of what percentage of the dataset is only labeled by this LF (coverage minus overlaps).

# %%
from snorkel.labeling.analysis import LFAnalysis

lf_names = [lf.name for lf in lfs]
LFAnalysis(L_dev).lf_summary(Y=Y_dev, lf_names=lf_names)

# %% [markdown]
# ### d) Balance accuracy/coverage

# %% [markdown]
# Often, by looking at the examples that an LF does and doesn't label, we can get ideas for how to improve it.
#
# The helper method `error_buckets()` groups examples by their predicted label and true label. For example, `buckets[(SPAM, HAM)]` contains the indices of data points that the LF labeled `SPAM` that actually belong to class `HAM`. This may give ideas for where the LF could be made more specific.

# %%
from snorkel.analysis.error_analysis import error_buckets

buckets = error_buckets(Y_dev, L_dev_array)

df_dev[["text", "label"]].iloc[buckets[(SPAM, HAM)]].head()

# %% [markdown]
# On the other hand, `buckets[(SPAM, SPAM)]` points to `SPAM` data points that the LF labeled correctly.

# %%
df_dev[["text", "label"]].iloc[buckets[(SPAM, SPAM)]].head()

# %% [markdown]
# And `buckets[(ABSTAIN, SPAM)]` points to data points that the LF abstained on that are actually `SPAM`.
# Many of these will be best captured by a separate LF, but browsing these examples can be a good check that your LF is capturing most of the examples that you intended it to.

# %%
df_dev[["text", "label"]].iloc[buckets[(ABSTAIN, SPAM)]].head()


# %% [markdown]
# Looking at all these examples, we notice that much of the time when "my" is used, it's referring to "my channel". We can update our LF to see how making this change affects accuracy and coverage.

# %%
@labeling_function()
def keywords_my_channel(x):
    return SPAM if "my channel" in x.text.lower() else ABSTAIN


lfs = [keywords_my_channel]
applier = PandasLFApplier(lfs)
L_dev = applier.apply(df_dev)

lf_names = [lf.name for lf in lfs]
LFAnalysis(L_dev).lf_summary(Y=Y_dev, lf_names=lf_names)

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
    return SPAM if "my" in x.text.lower() else ABSTAIN


@labeling_function()
def lf_subscribe(x):
    """Spam comments ask users to subscribe to their channels."""
    return SPAM if "subscribe" in x.text else ABSTAIN


@labeling_function()
def lf_link(x):
    """Spam comments post links to other channels."""
    return SPAM if "http" in x.text.lower() else ABSTAIN


@labeling_function()
def lf_please(x):
    """Spam comments make requests rather than commenting."""
    return (
        SPAM if any([word in x.text.lower() for word in ["please", "plz"]]) else ABSTAIN
    )


@labeling_function()
def lf_song(x):
    """Ham comments actually talk about the video's content."""
    return HAM if "song" in x.text.lower() else ABSTAIN


# %% [markdown]
# ### ii. Pattern-matching LFs (Regular Expressions)

# %% [markdown]
# If we want a little more control over a keyword search, we can look for regular expressions instead.

# %%
import re


@labeling_function()
def regex_check_out(x):
    """Spam comments say 'check out my video', 'check it out', etc."""
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN


# %% [markdown]
# ### iii.  Heuristic LFs

# %% [markdown]
# There may other heuristics or "rules of thumb" that you come up with as you look at the data.
# So long as you can express it in a function, it's a viable LF!

# %%
@labeling_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return HAM if len(x.text.split()) < 5 else ABSTAIN


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
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)


@labeling_function(preprocessors=[spacy])
def has_person(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN


# %% [markdown]
# Because spaCy is such a common preprocessor for NLP (Natural Language Processing) applications, we also provide an alias for a labeling_function that uses spaCy. This resulting LF is identical to the one defined manually above.

# %%
from snorkel.labeling.lf.nlp import nlp_labeling_function


@nlp_labeling_function()
def has_person_nlp(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN


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
    TextBlob(x.text).sentiment.polarity for i, x in df_dev.iterrows() if x.label == SPAM
]
ham_polarities = [
    TextBlob(x.text).sentiment.polarity for i, x in df_dev.iterrows() if x.label == HAM
]

plt.hist([spam_polarities, ham_polarities])

# %%
from textblob import TextBlob


@labeling_function()
def textblob_polarity(x):
    return HAM if TextBlob(x.text).sentiment.polarity > 0.3 else ABSTAIN


@labeling_function()
def textblob_subjectivity(x):
    return HAM if TextBlob(x.text).sentiment.subjectivity > 0.9 else ABSTAIN


# %% [markdown]
# ### Apply LFs

# %% [markdown]
# This tutorial demonstrates just a handful of the types of LFs that one might write for this task.
# Many of these are no doubt suboptimal.
# The strength of this approach, however, is that the LF abstraction provides a flexible interface for conveying a huge variety of supervision signals, and the `LabelModel` is able to denoise these signals, reducing the need for painstaking manual fine-tuning.

# %%
lfs = [
    keyword_my,
    lf_subscribe,
    lf_link,
    lf_please,
    lf_song,
    regex_check_out,
    short_comment,
    has_person_nlp,
    textblob_polarity,
    textblob_subjectivity,
]

# %% [markdown]
# With our full set of LFs, we can now apply these once again with `LFApplier` to get our the label matrices for the `train` and `dev` splits. We'll use the `train` split's label matrix to generate training labels with the Label Model. The `dev` split's label model is primarily helpful for looking at summary statistics.
#
# Note that the `pandas` format provides an easy interface that many practioners are familiar with, but it is also less optimized for scale. For larger datasets, more compute-intensive LFs, or larger LF sets, you may decide to use one of the other supported data formats such as `dask` or `spark` dataframes, and their corresponding applier objects.

# %%
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_dev = applier.apply(df_dev)
L_valid = applier.apply(df_valid)

lf_names = [lf.name for lf in lfs]
LFAnalysis(L_dev).lf_summary(Y=Y_dev, lf_names=lf_names)

# %% [markdown]
# We see that our labeling functions vary in coverage, accuracy, and how much they overlap/conflict with one another.
# We can view a histogram of how many weak labels the data points in our dev set have to get an idea of our total coverage.

# %%
import matplotlib.pyplot as plt


def plot_label_frequency(L):
    plt.hist(
        np.asarray((L != ABSTAIN).sum(axis=1)), density=True, bins=range(L.shape[1])
    )
    plt.xlabel("Number of labels")
    plt.ylabel("Fraction of dataset")


plot_label_frequency(L_train)

# %% [markdown]
# We see that over half of our training dataset data points have 0 or 1 weak labels.
# Fortunately, the signal we do have can be used to train a classifier with a larger feature set than just these labeling functions that we've created, allowing it to generalize beyond what we've specified.

# %% [markdown]
# ## 3. Combining Weak Labels with the Label Model

# %% [markdown]
# Our goal is now to convert these many weak labels into a single _noise-aware_ probabilistic (or confidence-weighted) label per data point.
# A simple baseline for doing this is to take the majority vote on a per-data point basis: if more LFs voted SPAM than HAM, label it SPAM (and vice versa).

# %%
from snorkel.labeling.model import MajorityLabelVoter

majority_model = MajorityLabelVoter()
Y_pred_train = majority_model.predict(L_train)
Y_pred_train

# %% [markdown]
# However, as we can clearly see by looking the summary statistics of our LFs in the previous section, they are not all equally accurate, and should ideally not be treated identically. In addition to having varied accuracies and coverages, LFs may be correlated, resulting in certain signals being overrepresented in a majority-vote-based model. To handle these issues appropriately, we will instead use a more sophisticated Snorkel `LabelModel` to combine our weak labels.
#
# This model will ultimately produce a single set of noise-aware training labels, which are probabilistic or confidence-weighted labels. We will then use these labels to train a classifier for our task. For more technical details of this overall approach, see our [NeurIPS 2016](https://arxiv.org/abs/1605.07723) and [AAAI 2019](https://arxiv.org/abs/1810.02840) papers.
#
# Note that no gold labels are used during the training process; the `LabelModel` is able to learn weights for the labeling functions using only the weak label matrix as input.

# %%
from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)

# %%
print(
    f"{'Majority Vote:':<14} {majority_model.score(L_valid, Y_valid)['accuracy']:0.3f}"
)
print(f"{'Label Model:':<14} {label_model.score(L_valid, Y_valid)['accuracy']:0.3f}")


# %% [markdown]
# While our `LabelModel` does improve over the majority vote baseline, it is still somewhat **limited as a classifier** for the reasons outlined in the Snorkel 101 guide.
#
# In the next section, we will use these training labels to train a discriminative classifier to see if we can improve performance further.

# %% [markdown]
# Before we do, we briefly confirm that the labels the `LabelModel` produces are probabilistic in nature.
# The following histogram shows the confidences we have that each data point has the label SPAM.
# The points we are least certain about will have labels close to 0.5.

# %%
def plot_probabilities_histogram(Y):
    plt.hist(Y, bins=10)
    plt.xlabel("Probability of SPAM")
    plt.ylabel("Number of Data Points")


Y_probs_train = label_model.predict_proba(L_train)
plot_probabilities_histogram(Y_probs_train[:, SPAM])

# %% [markdown]
# ## 4. Training a Classifier

# %% [markdown]
# In this final section of the tutorial, we'll use the noisy training labels we generated in the last section to train a classifier for our task.
#
# Note that because the output of the Snorkel `LabelModel` is just a set of labels, Snorkel easily integrates with most popular libraries for performing supervised learning: TensorFlow, Keras, PyTorch, Scikit-Learn, Ludwig, XGBoost, etc.
#
# In this tutorial we demonstrate using classifiers from Keras and Scikit-Learn.

# %% [markdown]
# For simlicity and speed, we use a simple "bag of n-grams" feature representation: each data point is represented by a one-hot vector marking which words or 2-word combinations are present in the comment text.

# %% [markdown]
# ### Featurization

# %%
from sklearn.feature_extraction.text import CountVectorizer

words_train = [row.text for i, row in df_train.iterrows()]
words_dev = [row.text for i, row in df_dev.iterrows()]
words_valid = [row.text for i, row in df_valid.iterrows()]
words_test = [row.text for i, row in df_test.iterrows()]

vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(words_train)
X_dev = vectorizer.transform(words_dev)
X_valid = vectorizer.transform(words_valid)
X_test = vectorizer.transform(words_test)

# %% [markdown]
# ### Filtering Unlabeled Data Points

# %% [markdown]
# As we saw earlier, some of the data points in our training set received no weak labels from our LFs.
# These examples are not helpful for training our classifier, as they convey no supervision signal, so we filter them out before training.

# %%
mask = L_train.sum(axis=1) != ABSTAIN * len(lfs)
X_train = X_train[mask, :]
Y_probs_train = Y_probs_train[mask]

# %% [markdown]
# ### Keras Classifier with Probabilistic Labels

# %% [markdown]
# Our Keras classifier is a simple logistic regression classifier.
# We compile it with a `categorical_crossentropy` loss so that it can handle probabilistic labels instead of integer labels.
# We use the common settings of an `Adam` optimizer and early stopping (evaluating the model on the validation set after each epoch and reloading the weights from when it achived the best score).

# %%
from snorkel.analysis.utils import probs_to_preds, preds_to_probs
from snorkel.analysis.metrics import metric_score
import tensorflow as tf

keras_model = tf.keras.Sequential()
keras_model.add(
    tf.keras.layers.Dense(
        2,
        input_dim=X_train.shape[1],
        activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
    )
)
optimizer = tf.keras.optimizers.Adam(lr=0.001)
keras_model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc", patience=10, verbose=1, restore_best_weights=True
)

keras_model.fit(
    X_train,
    Y_probs_train,
    validation_data=(X_valid, preds_to_probs(Y_valid, 2)),
    callbacks=[early_stopping],
    epochs=20,
    verbose=0,
)

Y_probs_test = keras_model.predict(X_test)
Y_preds_test = probs_to_preds(Y_probs_test)
print(f"Test Accuracy: {metric_score(Y_test, Y_preds_test, metric='accuracy')}")

# %% [markdown]
# Doing this, we observe an additional boost in accuracy over the `LabelModel` by multiple points---**the training set produced by the `LabelModel` successfully transferred our domain knowledge to the classifier, which was able to generalize beyond the noisy heuristics we provided in our LFs!**

# %% [markdown]
# We can compare this to the score we could have gotten if we had used our small labeled dev set directly as training data instead of using it to guide the creation of LFs.

# %%
keras_rounded_model = tf.keras.Sequential()
keras_rounded_model.add(
    tf.keras.layers.Dense(
        1,
        input_dim=X_train.shape[1],
        activation=tf.nn.sigmoid,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
    )
)
optimizer = tf.keras.optimizers.Adam(lr=0.001)
keras_rounded_model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)

keras_rounded_model.fit(
    X_dev,
    Y_dev,
    validation_data=(X_valid, Y_valid),
    callbacks=[early_stopping],
    epochs=20,
    verbose=0,
)

Y_probs_test = keras_rounded_model.predict(X_test)
Y_preds_test = np.round(Y_probs_test)
print(f"Test Accuracy: {metric_score(Y_test, Y_preds_test, metric='accuracy')}")

# %% [markdown]
# ### Scikit-Learn with Rounded Labels

# %% [markdown]
# If we want to use a library or model that doesn't accept probabilistic labels, we can replace each label distribution with the label of the class that has the maximum probability.
# This can easily be done using the helper method `probs_to_preds` (note, however, that this transformation is lossy, as we no longer have values for our confidence in each label).

# %%
Y_preds_train = probs_to_preds(Y_probs_train)

# %% [markdown]
# For example, this allows us to use standard models from Scikit-Learn.

# %%
from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression()
sklearn_model.fit(X_train, Y_preds_train)

sklearn_model.score(X_test, Y_test)

# %% [markdown]
# ## Summary

# %% [markdown]
# In this tutorial, we accomplished the following:
# * We introduced the concept of Labeling Functions (LFs) and demonstrated some of the forms they can take.
# * We used the Snorkel `LabelModel` to automatically learn how to combine many weak labels into strong probabilistic labels.
# * We showed that a classifier trained on a weakly supervised dataset can outperform an approach based on the LFs alone as it learns to generalize beyond the noisy heuristics we provide.

# %% [markdown]
# ### Next Steps

# %% [markdown]
# If you enjoyed this tutorial and you've already checked out the Snorkel 101 Guide, check out the `snorkel-tutorials` table of contents for other tutorials that you may find interesting, including demonstrations of how to use Snorkel for scene-graph detection (images), crowdsourcing, information extraction, data augmentation, and more. (TODO: Many links)
