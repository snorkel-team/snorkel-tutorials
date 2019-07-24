# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visual Relationship Detection
#
# In this tutorial, we focus on the task of classifying visual relationships. These are relationships among a pair of objects in images (e.g. "man riding bicycle"), where "man" and "bicycle" are the subject and object, respectively, and "riding" is the relationship predicate. 
#
# ![Visual Relationships](https://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.png)
#
# For the purpose of this tutorial, we operate over the [Visual Relationship Detection (VRD) dataset](https://cs.stanford.edu/people/ranjaykrishna/vrd/) and focus on action relationships. We define our three class classification task as **identifying whether a pair of bounding boxes represents the relationship of `ride`, `carry` or a different semantic relationship.**
#
# In the examples of the relationships shown below, the red box represents the _subject_ while the green box represents the _object_. The _predicate_ (e.g. kick) denotes what action connects the subject and the object.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ### 1. Load Dataset
# We load the VRD dataset and filter images with at least one semantic predicate in it. We load the train, valid, and test sets as Pandas DataFrame objects with the following fields:
# - label: The relationship between the objects. 0: `RIDE`, 1: `CARRY`, 2: `OTHER` action predicates
# - object_bbox: coordinates of the bounding box for the object `[ymin, ymax, xmin, xmax]`	
# - object_category: category of the object	
# - source_img: filename for image the relationship is in	
# - subject_bbox: coordinates of the bounding box for the object `[ymin, ymax, xmin, xmax]` 		
# - subject_category: category of the subject
#
# Note that the `train_df` object has a labels field with all -1s. This denotes the lack of labels for that particular dataset. In this tutorial, we will assign probabilistic labels to the training set by writing labeling functions over attributes of the subject and objects!

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

    if predicates[relationship['predicate']] == 'ride':
        new_relationship_dict['label'] = 0 
    elif predicates[relationship['predicate']] == 'carry':
        new_relationship_dict['label'] = 1
    else:
        new_relationship_dict['label'] = 2
        
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

valid_df.sample(n=35)

# %% [markdown]
# ## 2. Writing Labeling Functions
# We now write labeling functions to detect what relationship exists between pairs of bounding boxes. To do so, we can encode knowledge about the categories of subjects and objects usually involved in these relationships (e.g., `person` is usually the subject for predicates like `ride` and `carry`). We can encode common knowledge about these predicates, such as the subject is usually higher than the object for the predicate `ride` into the labeling functions. 

# %%
from snorkel.labeling.apply import PandasLFApplier
from snorkel.labeling.lf import labeling_function

RIDE = 0
CARRY = 1 
OTHER = 2
ABSTAIN = -1

# %%
lfs = []

#Category-based LFs
@labeling_function()
def LF_ride_object(x):
    if x.subject_category == 'person':
        if x.object_category in  ['bike', 'snowboard', 'motorcycle', 'horse']:
            return RIDE
    return ABSTAIN
lfs.append(LF_ride_object)

@labeling_function()
def LF_ride_rare_object(x):
    if x.subject_category == 'person':
        if x.object_category in ['bus', 'truck', 'elephant']:
            return RIDE
    return ABSTAIN
lfs.append(LF_ride_rare_object)

@labeling_function()
def LF_carry_object(x):
    if x.subject_category == 'person':
        if x.object_category in ['bag', 'surfboard', 'skis']:
            return CARRY
    return ABSTAIN
lfs.append(LF_carry_object)

@labeling_function()
def LF_carry_subject(x):
    if x.object_category == 'person':
        if x.subject_category in ['chair', 'bike', 'snowboard', 'motorcycle', 'horse']:
            return CARRY
    return ABSTAIN
lfs.append(LF_carry_subject)

@labeling_function()
def LF_person(x):
    if x.subject_category != 'person':
        return OTHER
    return ABSTAIN
lfs.append(LF_person)


#Distance-based LFs
@labeling_function()
def LF_ydist(x):
    if x.subject_bbox[3] < x.object_bbox[3]:
        return OTHER
    return ABSTAIN
lfs.append(LF_ydist)

@labeling_function()
def LF_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox)-np.array(x.object_bbox)) <= 1000:
        return OTHER
    return ABSTAIN
lfs.append(LF_dist)

#Size-based LF
@labeling_function()
def LF_area(x):
    subject_area = (x.subject_bbox[1] - x.subject_bbox[0])*(x.subject_bbox[3] - x.subject_bbox[2])
    object_area = (x.object_bbox[1] - x.object_bbox[0])*(x.object_bbox[3] - x.object_bbox[2])
    
    if subject_area/object_area <= 0.5:
        return OTHER
    return ABSTAIN
lfs.append(LF_area)

# %%
from snorkel.labeling.apply import PandasLFApplier

applier = PandasLFApplier(lfs)
L_train = applier.apply(train_df)
L_valid = applier.apply(valid_df)

# %% [markdown]
# Note that the labeling functions have varying empirical accuracies and coverages. Because of the imbalance in how the classes are defined, labeling functions that label the `OTHER` class have higher coverage than labeling functions for `RIDE` or `CARRY`. This reflects the distribution of classes in the dataset as well.

# %%
from snorkel.labeling.analysis import LFAnalysis

Y_valid = valid_df.label.values
lf_names= [lf.name for lf in lfs]

lf_analysis = LFAnalysis(L_valid)
lf_analysis.lf_summary(Y_valid, lf_names=lf_names)

# %% [markdown]
# ## 3. Train Label Model
# We now train a multi-class `LabelModel` to assign training labels to the unalabeled training set. 

# %%
from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train, seed=123, lr=0.01, log_freq=10, n_epochs=100)

# %%
label_model.score(L_valid, Y_valid, metrics=["accuracy"])

# %% [markdown]
# ## 4. Train a Classifier
# You can then use these training labels to train any standard discriminative model, such as [a state-of-the-art ResNet](https://github.com/KaimingHe/deep-residual-networks), which should learn to generalize beyond the LF's we've developed!

# %% [markdown]
# ## Make DataLoaders

# %%
TRAIN_DIR  = "data/VRD/sg_dataset/sg_train_images"
train_df['labels'] = label_model.predict(L_train)

# %%
# TODO: add bbox visualization

import os
from PIL import Image
import matplotlib.pyplot as plt

pos_idx = np.where(train_df["label"] == 0)[0]
for idx in pos_idx[:3]:
    img_path = os.path.join(TRAIN_DIR, train_df.iloc[idx].source_img)
    img = np.array(Image.open(img_path))
    plt.imshow(img)
    plt.show()

# %%
# TODO: add bbox visualization
neg_idx = np.where(train_df["label"] == 1)[0]
for idx in neg_idx[:3]:
    img_path = os.path.join(TRAIN_DIR, train_df.iloc[idx].source_img)
    img = np.array(Image.open(img_path))
    plt.imshow(img)
    plt.show()

# %%
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas
import torch
from torch import Tensor
from torchvision import transforms

from snorkel.classification.data import DictDataset, DictDataLoader, XDict, YDict

def union(bbox1, bbox2):
    """Creates the union of the two bboxes.
    Args:
        bbox1: y0, y1, x0, x1 format.
        bbox2: y0, y1, x0, x1 format.
    Returns:
        The union of the arguments.
    """
    y0 = min(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x0 = min(bbox1[2], bbox2[2])
    x1 = max(bbox1[3], bbox2[3])
    return [y0, y1, x0, x1]

def crop_img_arr(img_arr, bbox):
    """Crop img_arr [H x W x C] to bbox"""
    return img_arr[bbox[0]:bbox[1], bbox[2]:bbox[3], :]

class SceneGraphDataset(DictDataset):
    
    def __init__(self, name: str, split: str, image_dir: str, df: pandas.DataFrame, image_size=224) -> None:
        self.image_dir = Path(image_dir)
        X_dict = {
            "img_fn": df["source_img"].tolist(),
            "obj_bbox": df["object_bbox"].tolist(),
            "sub_bbox": df["subject_bbox"].tolist(),
            "obj_category": df["object_category"].tolist(),
            "sub_category": df["subject_category"].tolist()
        }
        Y_dict = {"scene_graph_task": torch.LongTensor(df["label"].to_numpy())} #change to take in the rounded train labels
        super(SceneGraphDataset, self).__init__(name, split, X_dict, Y_dict)
        
        # define standard set of transformations to apply to each image
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index: int) -> Tuple[XDict, YDict]:
        img_fn = self.X_dict["img_fn"][index]
        img_arr = np.array(Image.open(self.image_dir /  img_fn))


        obj_bbox = self.X_dict["obj_bbox"][index]
        sub_bbox = self.X_dict["sub_bbox"][index]
        obj_category = self.X_dict["obj_category"][index]
        sub_category = self.X_dict["sub_category"][index]

        # compute crops
        obj_crop = crop_img_arr(img_arr, obj_bbox)
        sub_crop = crop_img_arr(img_arr, sub_bbox)
        union_crop = crop_img_arr(img_arr, union(obj_bbox, sub_bbox))

        # transform each crop 
        x_dict = {
            "obj_crop": self.transform(Image.fromarray(obj_crop)),
            "sub_crop": self.transform(Image.fromarray(sub_crop)),
            "union_crop": self.transform(Image.fromarray(union_crop)),
            "obj_category": obj_category,
            "sub_category": sub_category
        }

        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        return x_dict, y_dict
    
    def __len__(self):
        return len(self.X_dict["img_fn"])
    
train_dl = DictDataLoader(
    SceneGraphDataset("train_dataset", "train", TRAIN_DIR, train_df),
    batch_size=16,
    shuffle=True
)

valid_dl = DictDataLoader(
    SceneGraphDataset("valid_dataset", "valid", TRAIN_DIR, valid_df),
    batch_size=16,
    shuffle=False
)

# %% [markdown]
# ## Build Model

# %%
import torch
import torch.nn as nn
import torchvision.models as models

def init_fc(layer):
    torch.nn.init.xavier_uniform_(fc.weight)
    fc.bias.data.fill_(0.01)

# initialize pretrained feature extractor
cnn = models.resnet18(pretrained=True)

# freeze the resnet weights
for param in cnn.parameters():
    param.requires_grad = False

in_features = cnn.fc.in_features
feature_extractor = nn.Sequential(*list(cnn.children())[:-1])

class FlatConcat(nn.Module):
    """Module that flattens and concatenates features"""
    def forward(self, *inputs):
        return torch.cat(
            [input.view(input.size(0), -1) for input in inputs],
            dim=1
        )


# %%
import pandas as pd
import csv

WEMB_SIZE = 100
class WordEmb(nn.Module):
    """Extract and concat word embeddings for obj and sub categories."""
    
    # ! bash scripts/get_glove.sh
    def __init__(self, glove_fn="data/glove/glove.6B.100d.txt"):
        super(WordEmb, self).__init__()

        self.word_embs = pd.read_csv(
            glove_fn, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
        )
    
    def _get_wordvec(self, word):
        return self.word_embs.loc[word].as_matrix()

    def forward(self, obj_category, sub_category):
        obj_emb = self._get_wordvec(obj_category)
        sub_emb = self._get_wordvec(sub_category)
        embs = np.concatenate([obj_emb, sub_emb], axis=1)
        return torch.FloatTensor(embs)


# %%
# initialize fully connected layer—maps 3 sets of image features to class logits
# includes 2x word embeddings for subj and obj category
fc = nn.Linear(in_features*3 + 2*WEMB_SIZE, 3)
init_fc(fc)

# %%
from snorkel.classification.task import Task, Operation

# %%
module_pool = nn.ModuleDict({
    "feat_extractor": feature_extractor,
    "prediction_head": fc,
    "feat_concat": FlatConcat(),
    "word_emb": WordEmb()
})

union_feat_op = Operation(
    name="union_feat_op",
    module_name="feat_extractor",
    inputs=[("_input_", "union_crop")]
)

sub_feat_op = Operation(
    name="sub_feat_op",
    module_name="feat_extractor",
    inputs=[("_input_", "sub_crop")]
)

obj_feat_op = Operation(
    name="obj_feat_op",
    module_name="feat_extractor",
    inputs=[("_input_", "obj_crop")]
)

word_emb_op = Operation(
    name="word_emb_op",
    module_name="word_emb",
    inputs=[("_input_", "sub_category"), ("_input_", "obj_category")]
)

concat_op = Operation(
    name="concat_op",
    module_name="feat_concat",
    inputs=[
        ("obj_feat_op", 0), 
        ("sub_feat_op", 0), 
        ("union_feat_op", 0), 
        ("word_emb_op", 0)
    ]
)

prediction_op = Operation(
    name="head_op",
    module_name="prediction_head",
    inputs=[("concat_op", 0)]
)

task_flow = [sub_feat_op, obj_feat_op, union_feat_op, word_emb_op, concat_op, prediction_op]

# %%
from functools import partial
from snorkel.classification.scorer import Scorer

import torch.nn.functional as F
def ce_loss(module_name, outputs, Y, active):
    # Subtract 1 from hard labels in Y to account for Snorkel reserving the label 0 for
    # abstains while F.cross_entropy() expects 0-indexed labels
    return F.cross_entropy(outputs[module_name][0][active], (Y.view(-1))[active])


def softmax(module_name, outputs):
    return F.softmax(outputs[module_name][0], dim=1)

pred_cls_task = Task(
    name="scene_graph_task",
    module_pool=module_pool,
    task_flow=task_flow,
    loss_func=partial(ce_loss, "head_op"),
    output_func=partial(softmax, "head_op"),
    scorer = Scorer(metrics=["accuracy"])
)

# %% [markdown]
# ## Train!

# %%
# %%time

from snorkel.classification.training import Trainer
from snorkel.classification.snorkel_classifier import SnorkelClassifier

model = SnorkelClassifier([pred_cls_task])
trainer = Trainer(
    n_epochs=1, lr=1e-3, 
    checkpointing=True, checkpointer_config={"checkpoint_dir": "checkpoint"}
)
trainer.train_model(model, [train_dl])

# %%
model.score([valid_dl])

# %%
