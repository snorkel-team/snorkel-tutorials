# %% [markdown]
# # Detecting spouse mentions in sentences

# %% [markdown]
# In this tutorial, we will see how Snorkel can be used for Information Extraction. We will walk through an example text classification task for information extraction, where we use labeling functions involving keywords and distant supervision.
# ### Classification Task
# <img src="imgs/sentence.jpg" width="700px;">
#
# We want to classify each __candidate__ or pair of people mentioned in a sentence, as being married at some point or not.
#
# In the above example, our candidate represents the possible relation `(Barack Obama, Michelle Obama)`. As readers, we know this mention is true due to external knowledge and the keyword of `wedding` occuring later in the sentence.
# We begin with some basic setup and data downloading.
#
# %%
# %matplotlib inline

import os
import pickle
import numpy as np

if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("spouse")

from utils import load_data

((df_dev, Y_dev), df_train, (df_test, Y_test)) = load_data()

# %% [markdown]
# **Input Data:** `df_dev`, `df_train`, and `df_test` are `Pandas DataFrame` objects, where each row represents a particular __candidate__. For our problem, a candidate consists of a sentence, and two people mentioned in the sentence. The DataFrames contain the fields `sentence`, which refers to the sentence of the candidate, `tokens`, the tokenized form of the sentence, and `person1_word_idx` and `person2_word_idx`, which represent `[start, end]` indices in the tokens at which the first and second person's name appear, respectively.
#
# We also have certain **preprocessed fields**, that we discuss a few cells below.

# %%
import pandas as pd

# Don't truncate text fields in the display
pd.set_option("display.max_colwidth", 0)

df_dev.head()

# %% [markdown]
# Let's look at a candidate in the development set:

# %%
from preprocessors import get_person_text

candidate = df_dev.loc[2]
person_names = get_person_text(candidate).person_names

print("Sentence: ", candidate["sentence"])
print("Person 1: ", person_names[0])
print("Person 2: ", person_names[1])

# %% [markdown]
# ### Preprocessing the Data
#
# In a real application, there is a lot of data preparation, parsing, and database loading that needs to be completed before we generate candidates and dive into writing labeling functions. Here we've pre-generated candidates in a pandas DataFrame object per split (train,dev,test).

# %% [markdown]
# ### Labeling Function Helpers
#
# When writing labeling functions, there are several functions you will use over and over again. In the case of text relation extraction as with this task, common functions include those for fetching text between mentions of the two people in a candidate, examing word windows around person mentions, and so on. We will wrap these functions as `preprocessors`.

# %%
from snorkel.preprocess import preprocessor


@preprocessor()
def get_text_between(cand):
    """
    Returns the text between the two person mentions in the sentence for a candidate
    """
    start = cand.person1_word_idx[1] + 1
    end = cand.person2_word_idx[0]
    cand.text_between = " ".join(cand.tokens[start:end])
    return cand


# %% [markdown]
# ### Candidate PreProcessors
#
# For the purposes of the tutorial, we have three fields (`between_tokens`, `person1_right_tokens`, `person2_right_tokens`) preprocessed in the data, which can be used when creating labeling functions. We also provide the following set of `preprocessor`s for this task in `preprocessors.py`, along with the fields these populate.
# * `get_person_text(cand)`: `person_names`
# * `get_person_lastnames(cand)`: `person_lastnames`
# * `get_left_tokens(cand)`: `person1_left_tokens`, `person2_left_tokens`

# %%
from preprocessors import get_left_tokens, get_person_last_names

POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1

# %%
from snorkel.labeling import labeling_function

# Check for the `spouse` words appearing between the person mentions
spouses = {"spouse", "wife", "husband", "ex-wife", "ex-husband"}


@labeling_function(resources=dict(spouses=spouses))
def lf_husband_wife(x, spouses):
    return POSITIVE if len(spouses.intersection(set(x.between_tokens))) > 0 else ABSTAIN


# %%
# Check for the `spouse` words appearing to the left of the person mentions
@labeling_function(resources=dict(spouses=spouses), pre=[get_left_tokens])
def lf_husband_wife_left_window(x, spouses):
    if len(set(spouses).intersection(set(x.person1_left_tokens))) > 0:
        return POSITIVE
    elif len(set(spouses).intersection(set(x.person2_left_tokens))) > 0:
        return POSITIVE
    else:
        return ABSTAIN


# %%
# Check for the person mentions having the same last name
@labeling_function(pre=[get_person_last_names])
def lf_same_last_name(x):
    p1_ln, p2_ln = x.person_lastnames

    if p1_ln and p2_ln and p1_ln == p2_ln:
        return POSITIVE
    return ABSTAIN


# %%
# Check for the word `married` between person mentions
@labeling_function()
def lf_married(x):
    return POSITIVE if "married" in x.between_tokens else ABSTAIN


# %%
# Check for words that refer to `family` relationships between and to the left of the person mentions
family = {
    "father",
    "mother",
    "sister",
    "brother",
    "son",
    "daughter",
    "grandfather",
    "grandmother",
    "uncle",
    "aunt",
    "cousin",
}
family = family.union({f + "-in-law" for f in family})


@labeling_function(resources=dict(family=family))
def lf_familial_relationship(x, family):
    return NEGATIVE if len(family.intersection(set(x.between_tokens))) > 0 else ABSTAIN


@labeling_function(resources=dict(family=family), pre=[get_left_tokens])
def lf_family_left_window(x, family):
    if len(set(family).intersection(set(x.person1_left_tokens))) > 0:
        return NEGATIVE
    elif len(set(family).intersection(set(x.person2_left_tokens))) > 0:
        return NEGATIVE
    else:
        return ABSTAIN


# %%
# Check for `other` relationship words between person mentions
other = {"boyfriend", "girlfriend", "boss", "employee", "secretary", "co-worker"}


@labeling_function(resources=dict(other=other))
def lf_other_relationship(x, other):
    return NEGATIVE if len(other.intersection(set(x.between_tokens))) > 0 else ABSTAIN


# %% [markdown]
# ### Distant Supervision Labeling Functions
#
# In addition to using factories that encode pattern matching heuristics, we can also write labeling functions that _distantly supervise_ examples. Here, we'll load in a list of known spouse pairs and check to see if the pair of persons in a candidate matches one of these.
#
# **DBpedia**
# http://wiki.dbpedia.org/
# Our database of known spouses comes from DBpedia, which is a community-driven resource similar to Wikipedia but for curating structured data. We'll use a preprocessed snapshot as our knowledge base for all labeling function development.
#
# We can look at some of the example entries from DBPedia and use them in a simple distant supervision labeling function.
#
# Make sure `dbpedia.pkl` is in the `spouse/data` directory.

# %%
with open("data/dbpedia.pkl", "rb") as f:
    known_spouses = pickle.load(f)

list(known_spouses)[0:5]


# %%
@labeling_function(resources=dict(known_spouses=known_spouses), pre=[get_person_text])
def lf_distant_supervision(x, known_spouses):
    p1, p2 = x.person_names
    if (p1, p2) in known_spouses or (p2, p1) in known_spouses:
        return POSITIVE
    else:
        return ABSTAIN


# %%
from preprocessors import last_name

# Last name pairs for known spouses
last_names = set(
    [
        (last_name(x), last_name(y))
        for x, y in known_spouses
        if last_name(x) and last_name(y)
    ]
)


@labeling_function(resources=dict(last_names=last_names), pre=[get_person_last_names])
def lf_distant_supervision_last_names(x, last_names):
    p1_ln, p2_ln = x.person_lastnames

    return (
        POSITIVE
        if (p1_ln != p2_ln)
        and ((p1_ln, p2_ln) in last_names or (p2_ln, p1_ln) in last_names)
        else ABSTAIN
    )


# %% [markdown]
# #### Apply Labeling Functions to the Data
# We create a list of labeling functions and apply them to the data

# %%
from snorkel.labeling import PandasLFApplier

lfs = [
    lf_husband_wife,
    lf_husband_wife_left_window,
    lf_same_last_name,
    lf_married,
    lf_familial_relationship,
    lf_family_left_window,
    lf_other_relationship,
    lf_distant_supervision,
    lf_distant_supervision_last_names,
]
applier = PandasLFApplier(lfs)

# %%
from snorkel.labeling import LFAnalysis

dev_L = applier.apply(df_dev)
train_L = applier.apply(df_train)

LFAnalysis(dev_L, lfs).lf_summary(Y_dev)

# %% [markdown]
# ### Training the Label Model
#
# Now, we'll train a model of the LFs to estimate their weights and combine their outputs. Once the model is trained, we can combine the outputs of the LFs into a single, noise-aware training label set for our extractor.

# %%
from snorkel.labeling import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(train_L, Y_dev, n_epochs=5000, log_freq=500, seed=12345)

# %% [markdown]
# ### Label Model Metrics
# Since our dataset is highly unbalanced (91% of the labels are negative), even a trivial baseline that always outputs negative can get a high accuracy. So we evaluate the label model using the F1 score and ROC-AUC rather than accuracy.

# %%
from snorkel.analysis import metric_score
from snorkel.utils import probs_to_preds

Y_probs_dev = label_model.predict_proba(dev_L)
Y_preds_dev = probs_to_preds(Y_probs_dev)
print(
    f"Label model f1 score: {metric_score(Y_dev, Y_preds_dev, probs=Y_probs_dev, metric='f1')}"
)
print(
    f"Label model roc-auc: {metric_score(Y_dev, Y_preds_dev, probs=Y_probs_dev, metric='roc_auc')}"
)

# %% [markdown]
# ### Part 4: Training our End Extraction Model
#
# In this final section of the tutorial, we'll use our noisy training labels alongside the development set labels to train our end machine learning model. We start by filtering out training examples which did not recieve a label from any LF, as these examples contain no signal. Then we concatenate them with dev set examples.
#
# %%
from snorkel.utils import preds_to_probs
from snorkel.labeling import filter_unlabeled_dataframe

# Change dev labels 1D array to 2D categorical labels array as required for training end model.
Y_probs_dev = preds_to_probs(Y_dev, 2)

Y_probs_train = label_model.predict_proba(train_L)
df_train_filtered, Y_probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=Y_probs_train, L=train_L
)

df_combined = pd.concat([df_dev, df_train_filtered])
Y_probs_combined = np.concatenate([Y_probs_dev, Y_probs_train_filtered], 0)

# %% [markdown]
# Next, we train a simple [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) network for classifying candidates. `tf_model` contains functions for processing features and building the keras model for training and evaluation.

# %%
from tf_model import get_model, get_feature_arrays

model = get_model()
tokens, idx1, idx2 = get_feature_arrays(df_combined)

batch_size = 64
num_epochs = 20  # TODO: Change this to ~50. Warning: Training takes several minutes!
model.fit(
    (tokens, idx1, idx2), Y_probs_combined, batch_size=batch_size, epochs=num_epochs
)

# %% [markdown]
# Finally, we evaluate the trained model by measuring its F1 score and ROC_AUC.

# %%
test_tokens, test_idx1, test_idx2 = get_feature_arrays(df_test)
probs = model.predict((test_tokens, test_idx1, test_idx2))
preds = probs_to_preds(probs)
print(
    f"Test F1 when trained with soft labels: {metric_score(Y_test, preds=preds, metric='f1')}"
)
print(
    f"Test ROC-AUC when trained with soft labels: {metric_score(Y_test, probs=probs, metric='roc_auc')}"
)

# %% [markdown]
# ## Summary
# In this tutorial, we showed how Snorkel can be used for Information Extraction. We demonstrated how to create LFs that leverage keywords and external knowledge bases (distant supervision). Finally, we showed how a model trained using the probabilistic outputs of the Label Model can achieve comparable performance while generalizing to all examples.
