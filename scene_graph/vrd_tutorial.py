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
# In this tutorial, we focus on the task of classifying visual relationships. For a given image, there might be many such relationships, defined formally as a `subject <predictate> object` (e.g. `person <riding> bike`).
#
# These are relationships among a pair of objects in images (e.g. "man riding bicycle"), where "man" and "bicycle" are the subject and object, respectively, and "riding" is the relationship predicate.
#
# ![Visual Relationships](https://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.png)
#
# For the purpose of this tutorial, we operate over the [Visual Relationship Detection (VRD) dataset](https://cs.stanford.edu/people/ranjaykrishna/vrd/) and focus on action relationships. We define our three class classification task as **identifying whether a pair of bounding boxes represents a particular relationship.**
#
# In the examples of the relationships shown below, the red box represents the _subject_ while the green box represents the _object_. The _predicate_ (e.g. kick) denotes what relationship connects the subject and the object.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np

# %% [markdown]
# ### 1. Load Dataset
# We load the VRD dataset and filter images with at least one action predicate in it, since these are more difficult to classify than geometric relationships like `above` or `next to`. We load the train, valid, and test sets as Pandas DataFrame objects with the following fields:
# - `label`: The relationship between the objects. 0: `RIDE`, 1: `CARRY`, 2: `OTHER` action predicates
# - `object_bbox`: coordinates of the bounding box for the object `[ymin, ymax, xmin, xmax]`
# - `object_category`: category of the object
# - `source_img`: filename for the corresponding image the relationship is in
# - `subject_bbox`: coordinates of the bounding box for the object `[ymin, ymax, xmin, xmax]`
# - `subject_category`: category of the subject
#
# Note that the `train_df` object has a labels field with all -1s. This denotes the lack of labels for that particular dataset. In this tutorial, we will assign probabilistic labels to the training set by writing labeling functions over attributes of the subject and objects!

# %%
import os

if os.path.basename(os.getcwd()) == "scene_graph":
    os.chdir("..")

# %%
from scene_graph.utils import load_vrd_data

train_df, valid_df, test_df = load_vrd_data()

print("Train Relationships: ", len(train_df))
print("Dev Relationships: ", len(valid_df))
print("Test Relationships: ", len(test_df))

# %% [markdown]
# ## 2. Writing Labeling Functions
# We now write labeling functions to detect what relationship exists between pairs of bounding boxes. To do so, we can encode knowledge about the categories of subjects and objects usually involved in these relationships (e.g., `person` is usually the subject for predicates like `ride` and `carry`). We can encode common knowledge about these predicates, such as the subject is usually higher than the object for the predicate `ride` into the labeling functions.

# %%
RIDE = 0
CARRY = 1
OTHER = 2
ABSTAIN = -1


from snorkel.labeling.lf import labeling_function

# Category-based LFs
@labeling_function()
def LF_ride_object(x):
    if x.subject_category == "person":
        if x.object_category in ["bike", "snowboard", "motorcycle", "horse"]:
            return RIDE
    return ABSTAIN


@labeling_function()
def LF_ride_rare_object(x):
    if x.subject_category == "person":
        if x.object_category in ["bus", "truck", "elephant"]:
            return RIDE
    return ABSTAIN


@labeling_function()
def LF_carry_object(x):
    if x.subject_category == "person":
        if x.object_category in ["bag", "surfboard", "skis"]:
            return CARRY
    return ABSTAIN


@labeling_function()
def LF_carry_subject(x):
    if x.object_category == "person":
        if x.subject_category in ["chair", "bike", "snowboard", "motorcycle", "horse"]:
            return CARRY
    return ABSTAIN


@labeling_function()
def LF_person(x):
    if x.subject_category != "person":
        return OTHER
    return ABSTAIN


# Distance-based LFs
@labeling_function()
def LF_ydist(x):
    if x.subject_bbox[3] < x.object_bbox[3]:
        return OTHER
    return ABSTAIN


@labeling_function()
def LF_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) <= 1000:
        return OTHER
    return ABSTAIN


# Size-based LF
@labeling_function()
def LF_area(x):
    subject_area = (x.subject_bbox[1] - x.subject_bbox[0]) * (
        x.subject_bbox[3] - x.subject_bbox[2]
    )
    object_area = (x.object_bbox[1] - x.object_bbox[0]) * (
        x.object_bbox[3] - x.object_bbox[2]
    )

    if subject_area / object_area <= 0.5:
        return OTHER
    return ABSTAIN


# %% [markdown]
# Note that the labeling functions have varying empirical accuracies and coverages. Because of the imbalance in how the classes are defined, labeling functions that label the `OTHER` class have higher coverage than labeling functions for `RIDE` or `CARRY`. This reflects the distribution of classes in the dataset as well.

# %%
from snorkel.labeling.apply import PandasLFApplier

lfs = [
    LF_ride_object,
    LF_ride_rare_object,
    LF_carry_object,
    LF_carry_subject,
    LF_person,
    LF_ydist,
    LF_dist,
    LF_area,
]

applier = PandasLFApplier(lfs)
L_train = applier.apply(train_df)
L_valid = applier.apply(valid_df)

# %%
from snorkel.labeling.analysis import LFAnalysis

Y_valid = valid_df.label.values
lf_names = [lf.name for lf in lfs]

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
label_model.score(L_valid, Y_valid, metrics=["f1_micro"])

# %% [markdown]
# ## 4. Train a Classifier
# You can then use these training labels to train any standard discriminative model, such as [a state-of-the-art ResNet](https://github.com/KaimingHe/deep-residual-networks), which should learn to generalize beyond the LF's we've developed!

# %% [markdown]
# #### Create DataLoaders for Classifier

# %%
from snorkel.classification.data import DictDataLoader
from scene_graph.model import FlatConcat, SceneGraphDataset, WordEmb, init_fc

# change to "scene_graph/data/VRD/sg_dataset/sg_train_images" for full set
TRAIN_DIR = "scene_graph/data/VRD/sg_dataset/samples"
train_df["labels"] = label_model.predict(L_train)

train_dl = DictDataLoader(
    SceneGraphDataset("train_dataset", "train", TRAIN_DIR, train_df),
    batch_size=16,
    shuffle=True,
)

valid_dl = DictDataLoader(
    SceneGraphDataset("valid_dataset", "valid", TRAIN_DIR, valid_df),
    batch_size=16,
    shuffle=False,
)

# %% [markdown]
# #### Define Loss Functions


# %%
import torch.nn.functional as F


def ce_loss(module_name, outputs, Y, active):
    return F.cross_entropy(outputs[module_name][0][active], (Y.view(-1))[active])


def softmax(module_name, outputs):
    return F.softmax(outputs[module_name][0], dim=1)


# %% [markdown]
# #### Define Model Architecture

# %%
import torchvision.models as models
import torch.nn as nn

from functools import partial
from snorkel.classification.scorer import Scorer
from snorkel.classification.task import Operation, Task

# initialize pretrained feature extractor
cnn = models.resnet18(pretrained=True)

# freeze the resnet weights
for param in cnn.parameters():
    param.requires_grad = False

# define input features
in_features = cnn.fc.in_features
feature_extractor = nn.Sequential(*list(cnn.children())[:-1])

# initialize FC layer: maps 3 sets of image features to class logits
WEMB_SIZE = 100
fc = nn.Linear(in_features * 3 + 2 * WEMB_SIZE, 3)
init_fc(fc)

# define layers
module_pool = nn.ModuleDict(
    {
        "feat_extractor": feature_extractor,
        "prediction_head": fc,
        "feat_concat": FlatConcat(),
        "word_emb": WordEmb(),
    }
)

union_feat_op = Operation(
    name="union_feat_op",
    module_name="feat_extractor",
    inputs=[("_input_", "union_crop")],
)

sub_feat_op = Operation(
    name="sub_feat_op", module_name="feat_extractor", inputs=[("_input_", "sub_crop")]
)

obj_feat_op = Operation(
    name="obj_feat_op", module_name="feat_extractor", inputs=[("_input_", "obj_crop")]
)

word_emb_op = Operation(
    name="word_emb_op",
    module_name="word_emb",
    inputs=[("_input_", "sub_category"), ("_input_", "obj_category")],
)

concat_op = Operation(
    name="concat_op",
    module_name="feat_concat",
    inputs=[
        ("obj_feat_op", 0),
        ("sub_feat_op", 0),
        ("union_feat_op", 0),
        ("word_emb_op", 0),
    ],
)

prediction_op = Operation(
    name="head_op", module_name="prediction_head", inputs=[("concat_op", 0)]
)

# define task flow through modules
task_flow = [
    sub_feat_op,
    obj_feat_op,
    union_feat_op,
    word_emb_op,
    concat_op,
    prediction_op,
]


pred_cls_task = Task(
    name="scene_graph_task",
    module_pool=module_pool,
    task_flow=task_flow,
    loss_func=partial(ce_loss, "head_op"),
    output_func=partial(softmax, "head_op"),
    scorer=Scorer(metrics=["f1_micro"]),
)

# %% [markdown]
# ### Train and Evaluate Model

# %%
from snorkel.classification.snorkel_classifier import SnorkelClassifier
from snorkel.classification.training import Trainer

model = SnorkelClassifier([pred_cls_task])
trainer = Trainer(
    n_epochs=1,
    lr=1e-3,
    checkpointing=True,
    checkpointer_config={"checkpoint_dir": "checkpoint"},
)
trainer.train_model(model, [train_dl])

# %%
model.score([valid_dl])
