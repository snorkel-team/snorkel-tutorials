# -*- coding: utf-8 -*-
# %% [markdown]
# # 🛠 Labeling Function Development in Snorkel

# %% [markdown]
# In the [Introductory Tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/spam_tutorial.ipynb),
# we created our training set using a wide range of labeling functions spanning pattern matching,
# weak supervision, and weak models.
# However, these LFs weren't the first ideas that popped into our heads.
# Typical LF development cycles include multiple iterations of ideation, refining, evaluation, and debugging.
# In this notebook, we walk through the development of two LFs using basic analysis tools in Snorkel.
#
# If you haven't done the [Introductory Tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/spam_tutorial.ipynb) yet, we recommend you do that first.
# We'll use the same [YouTube comments dataset](https://www.kaggle.com/goneee/youtube-spam-classifiedcomments) from Kaggle. Let's get to work!

# %% [markdown]
# ## Recommended practice for LF development
#
# Labeling function development is typically an iterative, creative, and free-form process.
# However, there are a few principles that we highly recommend using when developing labeling functions.
#
# * **Use the unlabeled training set for ideation whenever possible.** Reserve labeled examples for evaluation in order to avoid overfitting LFs or models.
# * **Develop labeling functions in isolation.** Avoid using a "boosting" approach, trying to develop LFs specifically to interact with behavior of existing LFs. If you use the existing LFs to identify uncovered areas of the data, generalize the observed pattern rather than writing an LF to cover those exact examples.
# * **Prioritize precision over coverage, up to 90% precision.** Precise LFs often result in higher performance models than ones with more coverage. However, coverage is often more difficult to obtain. So choose a 90% precision and 30% coverage version of an LF over a 95% precision and 10% coverage version.
#

# %% [markdown]
# ## Loading training and development set data
#
# In the [Introductory Tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/spam_tutorial.ipynb), we covered the key data set splits in Snorkel development.
# For LF development, we only need the training set (a large amount of unlabeled data) and the dev set (a large amount of labeled data).

# %%
import os

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("spam")

# %%
from utils import load_spam_dataset

df_train, df_dev, _, _ = load_spam_dataset()

# We pull out the label vectors for ease of use later
Y_dev = df_dev["label"].values


# Define label-to-int mapping for legibility
SPAM = 1
HAM = 0
ABSTAIN = -1


# %% [markdown]
# ## Initial ideation

# %% [markdown]
# Following recommended practice above, we'll start by looking at the training set to generate some ideas for LFs.

# %%
df_train[["author", "text", "video"]].sample(20, random_state=2)

# %% [markdown]
# One dominant pattern in the comments that look like spam is the use of the phrase "check out" (e.g. "check out my channel").
# Let's start with that.

# %% [markdown]
# ## Writing an LF to identify spammy comments that use the phrase "check out"

# %% [markdown]
# Let's start developing an LF to catch instances of commenters trying to get people to "check out" their channel, video, or website.
# We'll start by just looking for the exact string `"check out"` in the text, and compare that to looking for just `"check"` in the text.

# %%
from snorkel.labeling.lf import labeling_function


@labeling_function()
def check_out(x):
    return SPAM if "check out" in x.text.lower() else ABSTAIN


@labeling_function()
def check(x):
    return SPAM if "check" in x.text.lower() else ABSTAIN


# %% [markdown]
# Let's generate our label matrices and see how we do.

# %%
from snorkel.labeling.apply import PandasLFApplier

lfs = [check_out, check]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_dev = applier.apply(df_dev)

# %%
from snorkel.labeling.analysis import LFAnalysis

LFAnalysis(L_train, lfs).lf_summary().round(2)

# %%
LFAnalysis(L_dev, lfs).lf_summary(Y_dev).round(2)

# %% [markdown]
# So even these very simple rules do quite well!
# Following our principle above, we might want to pick the `check` rule, since both have high precision and `check` has higher coverage.
# But let's look at our error buckets to be sure.

# %%
from snorkel.analysis.error_analysis import error_buckets

buckets = error_buckets(Y_dev, L_dev[:, 1])
df_dev.iloc[buckets[(SPAM, HAM)]]

# %% [markdown]
# `check` is still looking good, since the false positive is specific to only a few of the most popular videos on YouTube (ones with billions of views).
# Now let's take a look at some places that `check` labeled `SPAM` on the training set to see if it matches our intuition or if we can identify some false positives.

# %%
df_train.iloc[L_train[:, 1] == SPAM].sample(10, random_state=1)

# %% [markdown]
# No clear false positives here, but many look like they could be labeled by `check_out` as well.
# Let's see where `check_out` abstained, but `check` labeled.

# %%
buckets = error_buckets(L_train[:, 0], L_train[:, 1])
df_train.iloc[buckets[(SPAM, ABSTAIN)]].sample(10, random_state=1)

# %% [markdown]
# Most of these seem like small modifications of "check out", like "check me out" or "check it out".
# Let's see if we can use regular expressions to account for this and get the best of both worlds.

# %%
import re


@labeling_function()
def regex_check_out(x):
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN


# %% [markdown]
# Again, let's generate our label matrices and see how we do.

# %%
lfs = [check_out, check, regex_check_out]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_dev = applier.apply(df_dev)

# %%
LFAnalysis(L_train, lfs).lf_summary().round(2)

# %%
LFAnalysis(L_dev, lfs).lf_summary(Y_dev).round(2)

# %% [markdown]
# We've split the difference in training set coverage, and increased our accuracy on the dev set to 100%!
# This looks promising.
# Let's verify that we corrected our false positive from before.

# %%
buckets = error_buckets(L_dev[:, 1], L_dev[:, 2])
df_dev.iloc[buckets[(ABSTAIN, SPAM)]]

# %% [markdown]
# To understand the coverage difference between `check` and `regex_check_out`, let's take a look at the training set.
# Remember: coverage isn't always good.
# Adding false positives will increase coverage.

# %%
buckets = error_buckets(L_train[:, 1], L_train[:, 2])
df_train.iloc[buckets[(ABSTAIN, SPAM)]].sample(10, random_state=1)

# %% [markdown]
# Most of these are SPAM, but a good number are false positives.
# **To keep precision high (while not sacrificing much in terms of coverage), we'd choose our regex-based rule.**

# %% [markdown]
# ## TextBlob

# %%
import matplotlib.pyplot as plt
from textblob import TextBlob


spam_polarities = [
    TextBlob(x.text).sentiment.polarity for _, x in df_dev.iterrows() if x.label == SPAM
]

ham_polarities = [
    TextBlob(x.text).sentiment.polarity for _, x in df_dev.iterrows() if x.label == HAM
]

plt.hist([spam_polarities, ham_polarities])
plt.title("TextBlob sentiment polarity scores")
plt.xlabel("Sentiment polarity score")
plt.ylabel("Count")
plt.legend(["Spam", "Ham"])
plt.show()

plt.hist([spam_polarities, ham_polarities], bins=[-0.5, 0, 0.5, 1])
plt.title("TextBlob sentiment polarity scores")
plt.xlabel("Sentiment polarity score")
plt.ylabel("Count")
plt.legend(["Spam", "Ham"])
plt.show()

# %%
spam_subjectivities = [
    TextBlob(x.text).sentiment.subjectivity
    for _, x in df_dev.iterrows()
    if x.label == SPAM
]

ham_subjectivities = [
    TextBlob(x.text).sentiment.subjectivity
    for _, x in df_dev.iterrows()
    if x.label == HAM
]

plt.hist([spam_subjectivities, ham_subjectivities])
plt.title("TextBlob sentiment subjectivity scores")
plt.xlabel("Sentiment subjectivity score")
plt.ylabel("Count")
plt.legend(["Spam", "Ham"])
plt.show()

plt.hist([spam_subjectivities, ham_subjectivities], bins=[0, 0.5, 1])
plt.title("TextBlob sentiment subjectivity scores")
plt.xlabel("Sentiment subjectivity score")
plt.ylabel("Count")
plt.legend(["Spam", "Ham"])
plt.show()

# %%
from snorkel.labeling.preprocess import preprocessor


@preprocessor()
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x


textblob_sentiment.memoize = True


@labeling_function(preprocessors=[textblob_sentiment])
def textblob_polarity_ham(x):
    return HAM if not (-0.5 < x.polarity < 0.5) else ABSTAIN


@labeling_function(preprocessors=[textblob_sentiment])
def textblob_polarity_spam(x):
    return SPAM if -0.5 < x.polarity < 0.5 else ABSTAIN


@labeling_function(preprocessors=[textblob_sentiment])
def textblob_subjectivity_ham(x):
    return HAM if x.subjectivity >= 0.5 else ABSTAIN


@labeling_function(preprocessors=[textblob_sentiment])
def textblob_subjectivity_spam(x):
    return SPAM if x.subjectivity < 0.5 else ABSTAIN


# %%
lfs = [
    textblob_polarity_ham,
    textblob_polarity_spam,
    textblob_subjectivity_ham,
    textblob_subjectivity_spam,
]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_dev = applier.apply(df_dev)

# %%
LFAnalysis(L_train, lfs).lf_summary().round(2)

# %%
LFAnalysis(L_dev, lfs).lf_summary(Y_dev).round(2)

# %%
