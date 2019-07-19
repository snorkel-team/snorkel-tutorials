# -*- coding: utf-8 -*-
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
# # Classifying Visual Relationships

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
import json

objects = json.load(open("./data/VRD/objects.json"))
predicates = json.load(open("./data/VRD/predicates.json"))
semantic_predicates = ['carry', 'cover', 'fly', 'look', 'lying on', 'park on', 'sit on', 'stand on', 'ride']


relationships_train = json.load(open("./data/VRD/annotations_train.json"))
relationships_test = json.load(open("./data/VRD/annotations_test.json"))

np.random.seed(123)
val_idx = list(np.random.choice(len(relationships_train), 1000, replace=False))
relationships_val = {key: value for i, (key, value) in enumerate(relationships_train.items()) if i in val_idx}
relationships_train = {key: value for i, (key, value) in enumerate(relationships_train.items()) if i not in val_idx}

print("Train Images: ", len(relationships_train))
print("Dev Images: ", len(relationships_val))
print("Test Images: ", len(relationships_test))


# %%
def flatten_vrd_relationship(img, relationship, objects, predicates):
    new_relationship_dict = {}
    new_relationship_dict['subject_category'] = objects[relationship['subject']['category']]
    new_relationship_dict['object_category'] = objects[relationship['object']['category']]
    new_relationship_dict['subject_bbox'] = relationship['subject']['bbox']
    new_relationship_dict['object_bbox'] = relationship['object']['bbox']

    new_relationship_dict['label'] = 1 if predicates[relationship['predicate']] == 'ride' else 2
    new_relationship_dict['source_img'] = img
    
    return new_relationship_dict


# %%
def vrd_to_pandas(relationships_set, objects, predicates, list_of_predicates=semantic_predicates):
    relationships = []

    for img in relationships_set:
        img_relationships = relationships_set[img]
        for relationship in img_relationships:
            predicate_idx = relationship['predicate']
            if predicates[predicate_idx] in list_of_predicates:
                relationships.append(flatten_vrd_relationship(img, relationship, objects, predicates))

    return pd.DataFrame.from_dict(relationships)


# %%
train_df = vrd_to_pandas(relationships_train, objects, predicates)
valid_df = vrd_to_pandas(relationships_val, objects, predicates)
test_df = vrd_to_pandas(relationships_test, objects, predicates)

print("Train Relationships: ", len(train_df))
print("Dev Relationships: ", len(valid_df))
print("Test Relationships: ", len(test_df))

valid_df.head()

# %%
# ride_keys = []
# other_keys = []

# for img in relationships_test:
#     img_relationships = relationships_test[img]
#     for relationship in img_relationships:
#         predicate_idx = relationship['predicate']
#         if predicates[predicate_idx] == 'ride':
#             ride_keys.append(img)
#         elif predicates[predicate_idx] in semantic_predicates:
#             other_keys.append(img)
            
# print("Number of ride relationships: ", len(ride_keys))
# print("Number of other relationships: ", len(other_keys))
# print("Percentage of Riding Relationships: ", len(ride_keys)/(len(ride_keys) + len(other_keys)))

# %%
from snorkel.labeling.apply import PandasLFApplier
from snorkel.labeling.lf import labeling_function

POS = 1
NEG = 2 
ABSTAIN = 0

# %%
lfs = []

#Category-based LFs
ride_objects = ['bike', 'snowboard', 'motorcycle', 'horse']
@labeling_function(resources=dict(ride_objects=ride_objects))
def LF_ride_object(x, ride_objects):
    if x.subject_category == 'person':
        if x.object_category in ride_objects:
            return POS
    return ABSTAIN
lfs.append(LF_ride_object)

rare_ride_objects = ['bus', 'truck', 'elephant']
@labeling_function(resources=dict(rare_ride_objects=rare_ride_objects))
def LF_ride_rare_object(x, rare_ride_objects):
    if x.subject_category == 'person':
        if x.object_category in rare_ride_objects:
            return POS
    return ABSTAIN
lfs.append(LF_ride_rare_object)

@labeling_function()
def LF_person(x):
    if x.subject_category != 'person':
        return NEG
    return ABSTAIN
lfs.append(LF_person)


#Distance-based LFs
#bbox in the form [ymin, ymax, xmin, xmax]
@labeling_function()
def LF_ydist(x):
    if x.subject_bbox[3] < x.object_bbox[3]:
        return NEG
    return ABSTAIN
lfs.append(LF_ydist)

@labeling_function()
def LF_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox)-np.array(x.object_bbox)) <= 1000:
        return NEG
    return ABSTAIN
lfs.append(LF_dist)

#Size-based LF
@labeling_function()
def LF_area(x):
    subject_area = (x.subject_bbox[1] - x.subject_bbox[0])*(x.subject_bbox[3] - x.subject_bbox[2])
    object_area = (x.object_bbox[1] - x.object_bbox[0])*(x.object_bbox[3] - x.object_bbox[2])
    
    if subject_area/object_area <= 0.5:
        return NEG
    return ABSTAIN
lfs.append(LF_area)

# %%
from snorkel.labeling.apply import PandasLFApplier

applier = PandasLFApplier(lfs)
L_train = applier.apply(train_df)
L_valid = applier.apply(valid_df)

# %%
from snorkel.analysis.utils import convert_labels
from snorkel.labeling.analysis import lf_summary

Y_valid = valid_df.label.values
lf_names= [lf.name for lf in lfs]
lf_summary(L_valid, Y_valid, lf_names=lf_names)

# %%
from snorkel.labeling.model import RandomVoter
from snorkel.analysis.metrics import metric_score
from snorkel.analysis.utils import probs_to_preds

rv_model = RandomVoter()
Y_probs_valid = rv_model.predict_proba(L_valid)
Y_preds_valid = probs_to_preds(Y_probs_valid)
metric_score(Y_valid, Y_preds_valid, probs=None, metric="f1")

# %%
from snorkel.labeling.model import MajorityClassVoter

mc_model = MajorityClassVoter()
mc_model.train_model(balance=[0.8, 0.2])
Y_probs_valid = mc_model.predict_proba(L_valid)
Y_preds_valid = probs_to_preds(Y_probs_valid)
metric_score(Y_valid, Y_preds_valid, probs=None, metric="f1")

# %%
from snorkel.labeling.model import MajorityLabelVoter

mv_model = MajorityLabelVoter()
Y_probs_valid = mv_model.predict_proba(L_valid)
Y_preds_valid = probs_to_preds(Y_probs_valid)
metric_score(Y_valid, Y_preds_valid, probs=None, metric="f1")

# %%
from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True, seed=123)
label_model.train_model(L_train, class_balance=[0.8, 0.2], seed=123, lr=0.005, log_train_every=10)

# %%
Y_probs_valid = label_model.predict_proba(L_valid)
Y_preds_valid = probs_to_preds(Y_probs_valid)
metric_score(Y_valid, Y_preds_valid, probs=None, metric="f1")

# %% [markdown]
# ## Now, train a discriminative model with your weak labels! 
# You can then use these training labels to train any standard discriminative model, such as [a state-of-the-art ResNet](https://github.com/KaimingHe/deep-residual-networks), which should learn to generalize beyond the LF's we've developed!
#
# The only change needed from standard procedure is to deal with the fact that the training labels Snorkel generates are _probabilistic_ (i.e. for the binary case, in [0,1])— luckily, this is a one-liner in most modern ML frameworks! For example, in TensorFlow, you can use the [cross-entropy loss](https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits).

# %%
