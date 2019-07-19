# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introductory Snorkel Tutorial: Spam Detection

# %% [markdown]
# In this tutorial, we will walk through the process of using `Snorkel` to classify YouTube comments as `spam` or `ham` (not spam). For an overview of Snorkel, visit [snorkel.stanford.edu](http://snorkel.stanford.edu). 
#
# For our task, we have access to a large amount of *unlabeled data*, which can be prohibitively expensive and slow to label manually. We therefore turn to weak supervision using *labeling functions*, or noisy, programmatic heuristics, to assign labels to unlabeled training data efficiently. We also have access to a small amount of labeled data, which we only for evaluation purposes. 
#
# The tutorial is divided in four parts:
# 1. **Loading Data**: We load a [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) from Kaggle.
#
# 2. **Writing Labeling Functions**: We write Python programs that take as input a datapoint and assign labels (or abstain) using heuristics, pattern matching, and third-party models.
#
# 3. **Combining Labels with the Label Model**: We use the outputs of the labeling functions over the train set as input to the LabelModel, which assings probabilistic labels to the train set.
#
# 4. **Training a Classifier**: We train a classifier that can predict labels for *any* YouTube comment using the probabilistic training labels from step 3.

# %% [markdown]
# ### Task: Spam Detection

# %% [markdown]
# We use a [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) that consists of YouTube comments from 5 videos. The task is to classify each comment as being `spam`, irrelevant or inappropriate messages, or `ham`, comments relevant to the video. 
#
# For example, the following comments are `spam`:
#
#         Subscribe to me for free Android games, apps..
#
#         Please check out my vidios
#
#         Subscribe to me and I'll subscribe back!!!
#
# and these are `ham`:
#
#
#         Came here to check the views, goodbye.
#
#         2 billion....Coming soon
#
#         This is a weird video.

# %% [markdown]
# ### Data Splits in Snorkel
#
# We split our data into 4 sets:
# * **Train Set**: This is the largest split of the dataset with no labels
# * **Validation (Valid) Set**: A labeled set used to tune hyperparameters for the end classifier. 
# * **Test Set**: A labeeld set used to evaluate the classifier. Note that users should not be looking at the test set and use it only for the final evaluation step.  
#
# While it is possible to write labeling functions without any labeled data and evaluate only with the classifier, ot os often usefule to have some labeled data to check performance against. We refer to this as the **development (dev) set**. This can either be a small labeled subset of the train set (e.g. 200 datapoints) or we can use the valid set as the dev set. Note that in the latter case, the our classifier can overfit to the valid set since both the labeling functions and the end model are tuned to the same labeled dataset.  

# %% [markdown]
# ## 1. Load data

# %% [markdown]
# We load the Kaggle dataset and create Pandas dataframe objects for each of the sets described above. Each dataframe consists of the following fields:
# * **COMMENT_ID**: Comment ID
# * **AUTHOR_ID**: Author ID
# * **DATE**: Date and time the comment was posted
# * **CONTENT**: The raw text content of the comment
# * **LABEL**: Whether the comment is `spam` (1) or `ham` (2)
# * **VIDEO_ID**: The video the comment is associated with
#
# The train and dev sets contain comments from 4 of the 5 videos in the dataset. The valid and test set both come from the last video and therefore belong to the same distribution, but are independent of the train and dev sets. The train set has `LABEL=0` for each comment, since it is unlabeled. 

# %%
from utils import load_spam_dataset

df_train, df_dev, df_valid, df_test = load_spam_dataset()


# %% [markdown]
# We look at some of the examples in the dev set here:

# %%
df_train.sample(5, random_state=1)

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

# %% [markdown]
# * Look at 10 examples; got any ideas?

# %%
# Don't truncate text fields in the display
pd.set_option('display.max_colwidth', 0)  

# Display just the text and label
df_dev[["CONTENT", "LABEL"]].sample(10, random_state=123)

# %% [markdown]
# The simplest way to create labeling functions in Snorkel is with the `@labeling_function()` decorator, which wraps a function for evaluating on a single `DataPoint` (in this case, a row of the dataframe).
#
# Looking at samples of our data, we see multiple messages where spammers are trying to get viewers to look at "my channel" or "my video," so we write a simple LF that labels an example as spam if it includes the word "my".

# %%
from snorkel.labeling.lf import labeling_function

# For clarity, we'll define constants to represent the class labels for spam, ham, and abstaining.
ABSTAIN = 0
SPAM = 1
HAM = 2

# We initialize an empty list that we'll add our LFs to as we create them
lfs = []

@labeling_function()
def keywords_my(x):
    return SPAM if 'my' in x.CONTENT.lower() else ABSTAIN

lfs.append(keywords_my)

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
# We can easily calculate the coverage of this LF (i.e., the percentage of the dataset that it labels) as follows:

# %%
coverage = L_train.nnz / L_train.shape[0]
print(f"Coverage: {coverage}")

# %% [markdown]
# To get an estimate of its accuracy, we can label the development set with it and compare that to the few gold labels we do have.

# %%
L_dev = applier.apply(df_dev)

# Note that we don't want to penalize the LF for examples where it abstained, 
# so we filter out both the predictions and the gold labels where the prediction
# is ABSTAIN
L_dev_array = np.asarray(L_dev.todense()).squeeze()
Y_dev_array = df_dev["LABEL"].values
accuracy = ((L_dev_array == Y_dev_array)[L_dev_array != ABSTAIN]).sum() / (L_dev_array != ABSTAIN).sum()
print(f"Accuracy: {accuracy}")

# %% [markdown]
# Alternatively, you can use the helper method `lf_summary` to report the following summary statistics:
# * Polarity: The set of labels this LF outputs
# * Coverage: The fraction of the dataset the LF labels
# * Overlaps: The fraction of the dataset where this LF and at least one other LF label
# * Conflicts: The fraction of the dataset where this LF and at least one other LF label and disagree
# * Correct: The number of `DataPoints` this LF labels correctly (if gold labels are provided)
# * Incorrect: The number of `DataPoints` this LF labels incorrectly (if gold labels are provided)
# * Emp. Acc.: The empirical accuracy of this LF (if gold labels are provided)

# %%
from snorkel.labeling.analysis import lf_summary

lf_names= [lf.name for lf in lfs]
lf_summary(L=L_dev, Y=Y_dev_array, lf_names=lf_names)

# %% [markdown]
# This LF is fairly accurate, but it only labels a fraction of the dataset.
# If we want to do well on our test set, we'll need more LFs.
#
# In the following subsections, we'll show just a few of the many types of LFs that you could write to generate a training dataset for this problem.

# %% [markdown]
# ### i. Keyword LFs

# %% [markdown]
# * Keywords

# %% [markdown]
# ### ii. Pattern-matching LFs

# %% [markdown]
# * Regexes

# %% [markdown]
# ### iii.  Heuristic LFs

# %% [markdown]
# * Length, early comma, etc.
# * SpaCy (preprocessor)

# %% [markdown]
# ### iv. Third-party Model LFs

# %% [markdown]
# * Sentiment classifier (preprocessor)

# %% [markdown]
# ### v. Write your own LFs

# %% [markdown]
# * Make a stub

# %% [markdown]
# ## 3. Combine with Label Model

# %% [markdown]
# * Pretty much copy prose from Spouse tutorial

# %% [markdown]
# * Run LabelModel, get probabilities
#     * Note: no labels are required or used
# * Look at probabilities (histogram)
# * What if we used this directly as a classifier? (score)
#     * Why we expect classifier we train to generalize better
#     * Look - we're randomly guessing on XX% of the data

# %% [markdown]
# * Can also compare to MV
#     * Does worse

# %% [markdown]
# ## 4. Predict with Classifier

# %% [markdown]
# * Now train classifier
#     * Can use any third-party classifier (plug into your existing pipelines!)
#     * Some libraries natively support probabilistic labels (us, TF); for others, can round.
# * Use bag-of-ngrams as features
# * [Train TF logreg w/ soft labels]
# * Score; see, we do better!
# * Also demonstrate sklearn logreg with hard labels (end model agnostic)
# * Compare with training on dev directly (see, we did better)
#     * And we could do even better with more raw unlabeled data

# %%
