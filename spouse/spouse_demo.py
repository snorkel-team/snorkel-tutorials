# ---
# jupyter:
#   jupytext:
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
# # Detecting spouse mentions in sentences

# %% [markdown]
# We will walk through an example text classification task to explore how Snorkel works with user-defined LFs. Run every cell in the notebook (unless otherwise noted) before proceeding to the next one! 
### Classification Task
# <img src="imgs/sentence.jpg" width="700px;">
#
# We want to classify each __candidate__ or pair of people mentioned in a sentence, as being married at some point or not.
#
# In the above example, our candidate represents the possible relation `(Barack Obama, Michelle Obama)`. As readers, we know this mention is true due to external knowledge and the keyword of `wedding` occuring later in the sentence.
# We begin with some basic setup and data downloading.
#
# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import pickle
import pandas as pd
import subprocess

subprocess.call(["bash", "download_data.sh"], shell=True)
with open('dev_data.pkl', 'rb') as f:
    dev_df = pickle.load(f)
    dev_labels = pickle.load(f)

with open('train_data.pkl', 'rb') as f:
    train_df = pickle.load(f)
# %% [markdown]
# **Input Data:** `dev_df` is a Pandas DataFrame object, where each row represents a particular __candidate__. The DataFrames contain the fields `sentence`, which refers to the sentence the candidate is in, `tokens`, the tokenized form of the sentence, `person1_word_idx` and `person2_word_idx`, which represent `[start, end]` indices in the tokens at which the first and second person's name appear, respectively. 
# 
# We also have certain **preprocessed fields**, that we discuss a few cells below. We have other tutorials focused on generating such datasets (e.g., from richy-formatted data), but assume we have access to a Pandas DataFrame for the purpose of this specific tutorial!

# %%
dev_df.head()

# %% [markdown]
# You'll interact with these candidates while writing labeling functions in Snorkel. We look at a candidate in the development set:

# %%
from preprocessors import get_person_text

candidate = dev_df.loc[112]
person_names = get_person_text(candidate)

print("Sentence:   \t", candidate['sentence'])
print("Person 1:   \t", person_names[0])
print("Person 2:   \t", person_names[1])

# %% [markdown]
## Part 2: Writing  Labeling Functions
# 
# In Snorkel, our primary interface through which we provide training signal to the end extraction model we are training is by writing **labeling functions (LFs)** (as opposed to hand-labeling massive training sets).  We'll go through some examples for our spouse classification task below.
# 
# A labeling function is a Python function that accepts a candidate, or a row of the DataFrame, as the input argument and outputs a label for the candidate. For ease of exposition in this notebook, we return `1` if it says the pair of persons in the candidate were married at some point,  `-1` if the pair of persons in the candidate were never married, and `0` if it doesn't know how to vote and abstains. In practice, many labeling functions are often unipolar: it labels only `1`s and `0`s, or it labels only `-1`s and `0`s.
# 
# (Note we will change our mapping to use `2` to represent the absence of a relationship to match the multiclass convention in the next notebook for the `LabelModel`. This does not affect this notebook.)
# 
# Recall that our goal is to ultimately train a high-performance classification model that predicts which of our candidates are true spouse relations. It turns out that we can do this by writing potentially low-quality labeling functions!

# %%
import re
import sys

import numpy as np
import scipy.sparse as sp

# %% [markdown]
##  I. Background

### Preprocessing the Database
# 
# In a real application, there is a lot of data preparation, parsing, and database loading that needs to be completed before we dive into writing labeling functions. Here we've pre-generated candidates in a pandas DataFrame object per split (train,dev,test).
# 
# ###  Using a _Development Set_ of Human-labeled Data
# 
# In our setting, we will use the phrase _development set_ to refer to a set of examples (here, a subset of our training set) which we label by hand and use to help us develop and refine labeling functions.  Unlike the _test set_, which we do not look at and use for final evaluation, we can inspect the development set while writing labeling functions. This is a list of `{-1,1}` labels.

# %% [markdown]
# ### Labeling Function Helpers
# 
# When writing labeling functions, there are several operators you will use over and over again. In the case of text relation extraction as with this task, common operators include fetching text between mentions of the two people in a candidate, examing word windows around person mentions, etc. Note that other domains and tasks, the required preprocessors will be different. 
# 
# We provide several helper functions in `spouse_preprocessors`:  these are Python helper functions that you can apply to candidates in the DataFrame to return objects that are helpful during LF development. You can (and should!) write your own helper functions to help write LFs.
# 
# We provide an example of a preprocessor definition here:

# %%
from snorkel.labeling.preprocess import preprocessor

@preprocessor
def get_text_between(cand):
    """
    Returns the text between the two person mentions in the sentence for a candidate
    """
    start = cand.person1_word_idx[1] + 1
    end = cand.person2_word_idx[0]
    cand.text_between = ' '.join(cand.tokens[start:end])
    return cand

# %% [markdown]
### Candidate PreProcessors
# 
# We provide a set of helper functions for this task in `preprocessors.py` that take as input a candidate, or row of a DataFrame in our case. For the purpose of the tutorial, we have two of these fields preprocessed in the data, which can be used when creating labeling functions.
# 
# `get_between_tokens(cand)`
# 
# `get_left_tokens(cand)`
# 
# `get_right_tokens(cand)`

# %% [markdown]
# II. Labeling Functions

## A. Pattern Matching Labeling Functions

# One powerful form of labeling function design is defining sets of keywords or regular expressions that, as a human labeler, you know are correlated with the true label. For example, we could define a dictionary of terms that occur between person names in a candidate. One simple dictionary of terms indicating a true relation could be, which we could use in a labeling function like shown below:
#     
#     spouses = {'spouse', 'wife', 'husband', 'ex-wife', 'ex-husband'}
# 
#  
#     @labeling_function(resources=dict(spouses=spouses), preprocessors=[get_left_tokens])
#     def LF_husband_wife_left_window(x, spouses):
#         if len(set(spouses).intersection(set(x.person1_left_tokens))) > 0:
#             return POS
#         elif len(set(spouses).intersection(set(x.person2_left_tokens))) > 0:
#             return POS
#         else:
#             return ABSTAIN
# 
# **Note that:**
# 1. To access the text between the person mentions, we can use the **`get_left_tokens` preprocessor!**
# 2. We use **resources like the spouses dictionary** to encode themes/categories of relationships!
# 
# There are a few advantages of having preprocessors and labeling functions in this form: 
# 
# **Data Agnostic:**  Operate over multiple data types without rewriting
#     
# **Incremental Processing:** Can create preprocessors as needed while writing LFs!
#      
# **Future Use:** Can store them for later for different tasks since they are reproducible and modular
#      
# **Optimizations:** Allows caching behind-the-scenes

# %%
from typing import List

from snorkel.labeling.apply import PandasLFApplier
from snorkel.labeling.lf import labeling_function
from snorkel.types import DataPoint

from preprocessors import get_left_tokens, get_person_last_names, get_person_text

POS = 1
NEG = -1 
ABSTAIN = 0 

# %%
# Check for the `spouse` words appearing between the person mentions
spouses = {'spouse', 'wife', 'husband', 'ex-wife', 'ex-husband'}
@labeling_function(resources=dict(spouses=spouses))
def LF_husband_wife(x, spouses):
    return POS if len(spouses.intersection(set(x.between_tokens))) > 0 else ABSTAIN

# %%
# Check for the `spouse` words appearing to the left of the person mentions
@labeling_function(resources=dict(spouses=spouses), preprocessors=[get_left_tokens])
def LF_husband_wife_left_window(x, spouses):
    if len(set(spouses).intersection(set(x.person1_left_tokens))) > 0:
        return POS
    elif len(set(spouses).intersection(set(x.person2_left_tokens))) > 0:
        return POS
    else:
        return ABSTAIN

# %%
# Check for the person mentions having the same last name
@labeling_function()
def LF_same_last_name(x):
    p1_ln, p2_ln = get_person_last_names(x)
    
    if p1_ln and p2_ln and p1_ln == p2_ln:
        return POS
    return ABSTAIN


# %%
# Check for the words `and ... married` between person mentions
@labeling_function()
def LF_and_married(x):
    return POS if 'and' in x.between_tokens and 'married' in x.person2_right_tokens else ABSTAIN

# %%
# Check for words that refer to `family` relationships between and to the left of the person mentions
family = ['father', 'mother', 'sister', 'brother', 'son', 'daughter',
              'grandfather', 'grandmother', 'uncle', 'aunt', 'cousin']
family = set(family+[f + '-in-law' for f in family])

@labeling_function(resources=dict(family=family))
def LF_familial_relationship(x, family):
    return POS if len(family.intersection(set(x.between_tokens))) > 0 else ABSTAIN  


@labeling_function(resources=dict(family=family), preprocessors=[get_left_tokens])
def LF_family_left_window(x, family):
    if len(set(family).intersection(set(x.person1_left_tokens))) > 0:
        return NEG
    elif len(set(family).intersection(set(x.person2_left_tokens))) > 0:
        return NEG
    else:
        return ABSTAIN

# %%
# Check for `other` relationship words between person mentions
other = {'boyfriend', 'girlfriend' 'boss', 'employee', 'secretary', 'co-worker'}
@labeling_function(resources=dict(other=other))
def LF_other_relationship(x, other):
    return NEG if len(other.intersection(set(x.between_tokens))) > 0 else ABSTAIN

# %% [markdown]
# #### Apply Labeling Functions to the Data
# We create a list of labeling functions and apply them to the data

# %%
lfs = [LF_husband_wife,
       LF_husband_wife_left_window,
       LF_same_last_name,
       LF_and_married, 
       LF_familial_relationship,
       LF_family_left_window,
       LF_other_relationship]
applier = PandasLFApplier(lfs)
L = applier.apply(dev_df)


# %% [markdown]
# ### Labeling Function Metrics
# 
# We can use the lf_summary function to measure various coverage related metrics for LFs. If we have gold labeled data, we can also evaluate accuracy.
# 
# #### Polarity
# The set of label values the LF can output when it doesn't abstain. It is common for each LF to have a single polarity.
# 
# #### Coverage
# The fraction of candidates that is labeled by our LF.
# 
# #### Overlaps
# The fraction of examples labeled by the LF that is also labeled by another LF.
# 
# #### Conflicts
# The fraction of examples labeled by the LF that is given a different (non-abstain) label by another LF.
# 
# #### Correct
# The number of correctly labeled examples on the gold labeled data.
# 
# #### Incorrect
# The number of incorrectly labeled examples on the gold labeled data.
# 
# #### Empirical Accuracy
# The fraction of correctly labeled examples on the gold data.

# %%
from snorkel.analysis.utils import convert_labels
from snorkel.labeling.analysis import lf_summary
from scipy.sparse import csr_matrix

Y_cat = convert_labels(dev_labels, "plusminus", "categorical")
L_cat = convert_labels(L.todense(), "plusminus", "categorical")
lf_names= [lf.name for lf in lfs]

lf_summary(csr_matrix(L_cat), Y_cat, lf_names=lf_names)

# %% [markdown]
# ## B. Distant Supervision Labeling Functions
# 
# In addition to using factories that encode pattern matching heuristics, we can also write labeling functions that _distantly supervise_ examples. Here, we'll load in a list of known spouse pairs and check to see if the pair of persons in a candidate matches one of these.
# 
# **DBpedia**
# http://wiki.dbpedia.org/
# Our database of known spouses comes from DBpedia, which is a community-driven resource similar to Wikipedia but for curating structured data. We'll use a preprocessed snapshot as our knowledge base for all labeling function development.
# 
# We can look at some of the example entries from DBPedia and use them in a simple distant supervision labeling function.
# 
# Make sure `dbpedia.pkl` is in the `tutorials/workshop/` directory. 

# %%
import pickle 

with open('dbpedia.pkl', 'rb') as f:
     known_spouses = pickle.load(f)
        
list(known_spouses)[0:5]

# %%
@labeling_function(resources=dict(known_spouses=known_spouses))
def LF_distant_supervision(x: DataPoint, known_spouses: List[str]) -> int:
    p1, p2 = get_person_text(x)
    return POS if (p1, p2) in known_spouses or (p2, p1) in known_spouses else ABSTAIN

# %%
# Helper function to get last name
def last_name(s):
    name_parts = s.split(' ')
    return name_parts[-1] if len(name_parts) > 1 else None 

# Last name pairs for known spouses
last_names = set([(last_name(x), last_name(y)) for x, y in known_spouses if last_name(x) and last_name(y)])

@labeling_function(resources=dict(last_names=last_names))
def LF_distant_supervision_last_names(x: DataPoint, last_names: List[str]) -> int:
    p1_ln, p2_ln = get_person_last_names(x)
    
    return POS if (p1_ln != p2_ln) and ((p1_ln, p2_ln) in last_names or (p2_ln, p1_ln) in last_names) else ABSTAIN

# %% [markdown]
# Every time you write a new labeling function, add it to appliers and make sure to include it in the new L matrix!

# %%
applier = PandasLFApplier([LF_husband_wife,
                           LF_husband_wife_left_window,
                           LF_same_last_name,
                           LF_and_married, 
                           LF_familial_relationship,
                           LF_family_left_window,
                           LF_other_relationship,
                           LF_distant_supervision,
                           LF_distant_supervision_last_names])

# %%
dev_L = applier.apply(dev_df)
with open('dev_L.pkl', 'wb') as f:
    pickle.dump(dev_L, f)
    
train_L = applier.apply(train_df)
with open('train_L.pkl', 'wb') as f:
    pickle.dump(train_L, f)

# %% [markdown]
# ## C. Writing Custom Labeling Functions
# 
# The strength of LFs is that you can write any arbitrary function and use it to supervise a classification task. This approach can combine many of the same strategies discussed above or encode other information. 
# 
# For example, we observe that when mentions of person names occur far apart in a sentence, this is a good indicator that the candidate's label is False. You can write a labeling function that uses preprocessor `get_text_between` or the existing field `get_between_tokens` to write such an LF!
# 
# 
#     
# **IMPORTANT** Good labeling functions manage a trade-off between high coverage and high precision. When constructing your dictionaries, think about building larger, noiser sets of terms instead of relying on 1 or 2 keywords. Sometimes a single word can be very predictive (e.g., `ex-wife`) but it's almost always better to define something more general, such as a regular expression pattern capturing _any_ string with the `ex-` prefix. 
# 
# **Try editing and running the cells below!**

# %%
# @labeling_function()
# def LF_new(x: DataPoint) -> int:
#     return POS if x.person1_word_idx[0] > 3 else ABSTAIN #TODO: Change this!

# applier = PandasLFApplier([LF_new])

# %%
# new_dev_L = applier.apply(dev_df)
# sp.hstack((dev_L, new_dev_L), format='csr')
# with open('dev_L.pkl', 'wb') as f:
#     pickle.dump(dev_L, f)
    
# new_train_L = applier.apply(train_df)
# sp.hstack((train_L, new_train_L), format='csr')
# with open('train_L.pkl', 'wb') as f:
#     pickle.dump(train_L, f)
