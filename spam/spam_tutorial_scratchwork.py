#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'spam'))
	print(os.getcwd())
except:
	pass

#%%
from utils import load_spam_dataset

df_train, df_dev, df_valid, df_test = load_spam_dataset()


#%%
df_valid.sample(5)

#%% [markdown]
# ### Test end model quality

#%%
# from ludwig.api import LudwigModel

# df_train['LABEL'] = df_train['LABEL'].map({1: 1, 2: 0})
# df_valid['LABEL'] = df_valid['LABEL'].map({1: 1, 2: 0})
# df_test['LABEL'] = df_test['LABEL'].map({1: 1, 2: 0})


#%%
# model_definition = {
#     "input_features": [{"name": "CONTENT", "type": "text"}],
#     "output_features": [{"name": "LABEL", "type": "binary"}],
#     "training": {"epochs": 1},
# }
# model = LudwigModel(model_definition)
# train_stats = model.train(
#     data_train_df=df_train,
#     data_validation_df=df_valid,
#     data_test_df=df_test,
#     logging_level=20,
#     epochs=4,
# )
# train_stats

#%% [markdown]
# ### Test LF ideas

#%%
from snorkel.labeling.lf import labeling_function

lfs = []

@labeling_function()
def lf_subscribe(x):
    return 1 if "subscribe" in x.CONTENT else 0
lfs.append(lf_subscribe)

@labeling_function()
def lf_check_out(x):
    return 1 if "check" in x.CONTENT.lower() and "out" in x.CONTENT.lower() else 0
lfs.append(lf_check_out)

@labeling_function()
def lf_my(x):
    return 1 if "my" in x.CONTENT.lower() else 0
lfs.append(lf_my)

@labeling_function()
def lf_link(x):
    return 1 if "http" in x.CONTENT.lower() else 0
lfs.append(lf_link)

@labeling_function()
def lf_please(x):
    return 1 if "please" in x.CONTENT.lower() else 0
lfs.append(lf_please)

@labeling_function()
def lf_come(x):
    return 1 if "come" in x.CONTENT.lower() else 0
lfs.append(lf_come)

@labeling_function()
def lf_song(x):
    return 2 if "song" in x.CONTENT.lower() else 0
lfs.append(lf_song)

@labeling_function()
def lf_short(x):
    return 2 if len(x.CONTENT.split()) < 5 else 0
lfs.append(lf_short)


#%%
from textblob import TextBlob

@labeling_function(resources={"threshold": 0.3})
def lf_polarity(x, threshold):
    return 2 if TextBlob(x.CONTENT).sentiment.polarity > threshold else 0
lfs.append(lf_polarity)

@labeling_function(resources={"threshold": 0.9})
def lf_subjectivity(x, threshold):
    return 2 if TextBlob(x.CONTENT).sentiment.subjectivity > threshold else 0
lfs.append(lf_subjectivity)


#%%
import re
def find_urls(string):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return urls

@labeling_function()
def lf_urls(x):
    return 1 if len(find_urls(x.CONTENT)) > 0 else 0
lfs.append(lf_urls)


#%%
#PV LFs
from snorkel.labeling.preprocess.nlp import SpacyPreprocessor
spacy = SpacyPreprocessor(text_field="CONTENT", doc_field="doc")
@labeling_function(preprocessors=[spacy])
def lf_number(x):
    num_numerals = 0
    tokens = x.doc
    for i in range(len(tokens)):
        if tokens[i].pos_ == "NUM":
            num_numerals+=1

    if num_numerals == 0:
        return 2
    else:
        return 0
lfs.append(lf_number)

@labeling_function(preprocessors=[spacy])
def lf_names(x):
    num_names = 0
    tokens = x.doc
    for i in range(len(tokens)):
        if tokens[i].pos_ == "PROPN":
            num_names+=1

    if num_names < 1:
        return 2
    else:
        return 0
lfs.append(lf_names)

@labeling_function()
def lf_have(x):
    return 1 if "have you" in x.CONTENT.lower() else 0
lfs.append(lf_have)


#%%
@labeling_function()
def early_comma(x):
    words = x.CONTENT.split()
    for word in words[:3]:
        if "," in word:
            return 1
    return 0
lfs.append(early_comma)


@labeling_function()
def word_lengths(x):
    words = x.CONTENT.split()
    lengths = [len(word) for word in words]
    mean = sum(lengths) / len(lengths)
    return 2 if mean < 4 else 0
lfs.append(word_lengths)


#%%
from snorkel.labeling.preprocess.nlp import SpacyPreprocessor

spacy_preprocessor = SpacyPreprocessor(text_field="CONTENT", doc_field="doc", disable=[])
spacy_preprocessor.memoize = True

spam_words = {"check", "hey", "hi", "guys", "my", "channel", "subscribe", "like"}

def has_spam_words(x):
    return len(spam_words.intersection(t.lemma_ for t in x.doc)) > 0


def is_long(x, t=20):
    return len(x.doc) > t


@labeling_function(preprocessors=[spacy_preprocessor])
def lf_ner_spam(x):
    if not (has_spam_words(x) or is_long(x)):
        return 0
    for ent in x.doc.ents:
        if ent.label_ == "PERSON":
            return 1
    return 0
lfs.append(lf_ner_spam)


@labeling_function(preprocessors=[spacy_preprocessor])
def lf_ner_ham(x):
    if has_spam_words(x) or is_long(x):
        return 0
    for ent in x.doc.ents:
        if ent.label_ == "PERSON":
            return 2
    return 0
lfs.append(lf_ner_ham)


#%%
# from textblob import TextBlob
# from collections import defaultdict

# scores = defaultdict(list)
# sentiments = []
# for i, x in df_valid.iterrows():
#     sentiment = TextBlob(x.CONTENT).sentiment.subjectivity
#     sentiments.append(sentiment)
#     scores[x.LABEL].append(sentiment)

# print(np.mean(scores[1]), np.mean(scores[2]))
# plt.hist(scores.values())


#%%
from snorkel.labeling.apply import PandasLFApplier

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_valid = applier.apply(df_valid)


#%%
from snorkel.analysis.utils import convert_labels
from snorkel.labeling.analysis import lf_summary

Y_valid = convert_labels(df_valid.LABEL.values, "onezero", "categorical")
lf_names= [lf.name for lf in lfs]
lf_summary(L_valid, Y_valid, lf_names=lf_names)


#%%
import matplotlib.pyplot as plt
# View label frequency
plt.hist(np.asarray((L_train != 0).sum(axis=1)), density=True, bins=range(len(lfs)))


#%%
from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.train_model(L_train)


#%%
from snorkel.analysis.metrics import metric_score
from snorkel.analysis.utils import probs_to_preds

Y_probs_valid = label_model.predict_proba(L_valid.todense())
Y_preds_valid = probs_to_preds(Y_probs_valid)
metric_score(Y_valid, Y_preds_valid, probs=None, metric="accuracy")


#%%
# View probabilities distribution
plt.hist(np.stack((Y_probs_valid[:, 0], Y_valid-1), axis=1), density=True)

#%% [markdown]
# ### Compare MV

#%%
from snorkel.labeling.model import MajorityLabelVoter

mv_model = MajorityLabelVoter()
Y_probs_valid = mv_model.predict_proba(L_valid.todense())
Y_preds_valid = probs_to_preds(Y_probs_valid)
metric_score(Y_valid, Y_preds_valid, probs=None, metric="accuracy")


#%%
# For testing MV->EM
Y_probs_train = mv_model.predict_proba(L_train.todense())

#%% [markdown]
# ### Ludwig with Weak Labels (Rounded)

#%%
# Don't train on examples with no labels
mask = L_train.sum(axis=1) > 0
L_train_filtered = L_train[mask.nonzero()[0], :]
df_train_filtered = df_train[mask]


#%%
# Y_probs_train = label_model.predict_proba(L_train_filtered.todense())
Y_preds_train = probs_to_preds(Y_probs_train)


#%%
df_train_filtered['LABEL'] = convert_labels(Y_preds_train, "categorical", "onezero")
df_valid['LABEL'] = df_valid['LABEL'].map({1: 1, 2: 0})
df_test['LABEL'] = df_test['LABEL'].map({1: 1, 2: 0})


#%%
from ludwig.api import LudwigModel

model_definition = {
    "input_features": [{"name": "CONTENT", "type": "text"}],
    "output_features": [{"name": "LABEL", "type": "binary"}],
    "training": {"epochs": 6},
}
ludwig_model = LudwigModel(model_definition)

train_stats = ludwig_model.train(
    data_train_df=df_train_filtered,
    data_validation_df=df_valid,
    data_test_df=df_test,
    logging_level=20,
)
train_stats


#%%
Y_preds_valid = convert_labels(ludwig_model.predict(df_valid).LABEL_predictions.values.astype(int), "onezero", "categorical")
metric_score(Y_valid, Y_preds_valid, probs=None, metric="accuracy")

#%% [markdown]
# ## Compare on Test Set

#%%
L_test = applier.apply(df_test)
Y_test = convert_labels(df_test.LABEL.values, "onezero", "categorical")

#%% [markdown]
# ### MV

#%%
from snorkel.labeling.model import MajorityLabelVoter

mv_model = MajorityLabelVoter()
# mv_model.score(L_test, Y_test)
Y_probs_test = mv_model.predict_proba(L_test.todense())
Y_preds_test = probs_to_preds(Y_probs_test)
metric_score(Y_test, Y_preds_test, probs=None, metric="accuracy")

#%% [markdown]
# ### LabelModel

#%%
from snorkel.analysis.metrics import metric_score
from snorkel.analysis.utils import probs_to_preds

# label_model.score(L_test, Y_test)
Y_probs_test = label_model.predict_proba(L_test.todense())
Y_preds_test = probs_to_preds(Y_probs_test)
metric_score(Y_test, Y_preds_test, probs=None, metric="accuracy")

#%% [markdown]
# ### EndModel (Ludwig)

#%%
Y_preds_test = convert_labels(ludwig_model.predict(df_test).LABEL_predictions.values.astype(int), "onezero", "categorical")
metric_score(Y_test, Y_preds_test, probs=None, metric="accuracy")


#%%



