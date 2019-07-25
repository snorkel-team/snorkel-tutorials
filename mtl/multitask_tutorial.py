# -*- coding: utf-8 -*-
# %% [markdown]
# # Multi-Task Learning (MTL) Basics Tutorial

# %% [markdown]
# Multi-task learning is becoming a standard tool for the modern ML practioner. A major requirement of this approach is the ability to easily *add new datasets, label sets, tasks, and metrics* (and just as easily remove them). Thus, in the Snorkel multi-task model design, each of these concepts have been decoupled.

# %% [markdown]
# The purpose of this tutorial is to introduce the basic interfaces and flow of the multi-task learning tools within Snorkel (we assume that you have prior experience with MTL, so we don't motivate or explain multi-task learning at large here.
#
# In this notebook, we'll look at a simple MTL model with only two tasks, each having distinct data and only one set of ground truth labels ("gold" labels). We'll also use data where the raw data is directly usable as features, for simplicity (i.e., unlike text data, where we would first need to tokenize and transform the data into token ids).

# %% [markdown]
# ## Environment Setup

# %%
# %matplotlib inline

from snorkel.analysis.utils import set_seed

SEED = 123
set_seed(SEED)

# %% [markdown]
# ## Create Toy Data

# %% [markdown]
# We'll now create a toy dataset to work with.
# Our data points are 2D points in a square centered on the origin.
# Our tasks will be classifying whether these points are:
#
# 1. Inside a **unit circle** centered on the origin
# 2. Inside a **unit square** centered on the origin
#
# We'll visualize these decision boundaries in a few cells.
#
# _Note: In this toy example, we don't expect these specific tasks to help each other learn, but this is often a benefit of joint training in MTL settings._

# %%
import numpy as np

N = 500  # Data points per dataset
R = 1  # Unit distance

# Dataset 0 ("circle")
circle_data = np.random.uniform(0, 1, size=(N, 2)) * 2 - 1
circle_labels = circle_data[:, 0] ** 2 + circle_data[:, 1] ** 2 < R

# Dataset 1 ("square")
square_data = np.random.uniform(0, 1, size=(N, 2)) * 2 - 1
square_labels = (abs(square_data[:, 0]) < 0.5) * (abs(square_data[:, 1]) < 0.5)

# %% [markdown]
# Note that, as is the case throughout the Snorkel repo, the **label -1 is reserved for abstaining/no label**; all actual labels have non-negative integer values: 0, 1, 2, .... This provides flexibility for supervision sources to label only portions of a dataset, for example.

# %% [markdown]
# And we can view the ground truth labels of our tasks visually to confirm our intuition on what the decision boundaries look like.

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)

axs[0].scatter(circle_data[:, 0], circle_data[:, 1], c=circle_labels)
axs[0].set_aspect("equal", "box")
axs[0].set_title("Task0", fontsize=10)

axs[1].scatter(square_data[:, 0], square_data[:, 1], c=square_labels)
axs[1].set_aspect("equal", "box")
axs[1].set_title("Task1", fontsize=10)

plt.show()

# %% [markdown]
# Here, we wrap the `train_test_split` function to split the data into train/valid/test splits.

# %%
from sklearn.model_selection import train_test_split


def split_data(data, labels, seed=123):
    """Split data twice using sklearn train_test_split helper."""

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED
    )

    return (
        {"train": X_train, "valid": X_val, "test": X_test},
        {"train": y_train, "valid": y_val, "test": y_test},
    )


# %% [markdown]
# We'll now divide the synthetic `circle_data` and `square_data` into train/valid/test splits using our helper function `split_data()`.

# %%
circle_data_splits, circle_label_splits = split_data(
    circle_data, circle_labels, seed=SEED
)
square_data_splits, square_label_splits = split_data(
    square_data, square_labels, seed=SEED
)

# %% [markdown]
# ## Make DataLoaders

# %% [markdown]
# With our data now loaded/created, we can now package it up into `DictDataset`s for training. This object is a simple wrapper around `torch.utils.data.Dataset` and stores data fields and labels as dictionaries.
#
# In the `DictDataset`, each label corresponds to a particular `Task` by name.  We'll define these `Task` objects in the following section as we define our model.
#
# `DictDataloader` is a wrapper for `torch.utils.data.Dataloader`, which handles the collate function for `DictDataset` appropriately.

# %%
import torch
from snorkel.classification.data import DictDataset, DictDataLoader

dataloaders = []
for split in ["train", "valid", "test"]:
    X_dict = {"circle_data": torch.FloatTensor(circle_data_splits[split])}
    Y_dict = {"circle_task": torch.LongTensor(circle_label_splits[split])}
    dataset = DictDataset("Circle", split, X_dict, Y_dict)
    dataloader = DictDataLoader(dataset, batch_size=32)
    dataloaders.append(dataloader)

for split in ["train", "valid", "test"]:
    X_dict = {"square_data": torch.FloatTensor(square_data_splits[split])}
    Y_dict = {"square_task": torch.LongTensor(square_label_splits[split])}
    dataset = DictDataset("Square", split, X_dict, Y_dict)
    dataloader = DictDataLoader(dataset, batch_size=32)
    dataloaders.append(dataloader)

# %% [markdown]
# We now have 6 data loaders, one for each task (`circle_task` and `square_task`) for each split (`train`, `valid`, `test`).

# %% [markdown]
# ## Define Model

# %% [markdown]
# Now we'll define the `SnorkelClassifier`, which is build from a list of `Tasks`.

# %% [markdown]
# ### Tasks

# %% [markdown]
# A `Task` represents a path through a neural network. In `SnorkelClassifier`, this path corresponds to a particular sequence of PyTorch modules through which each example will make a forward pass.
#
# To specify this sequence of modules, each `Task` defines a **module pool** (a set of modules that it relies on) and a **task flow**—a sequence of `Operation`s. Each `Operation` defines a module and the inputs to feed to that module. These inputs are described with a list of tuples, where each tuple is either (`_input_`, \[field_name\]), or the name of a previous operation and the index of its output to use (most modules have only a single output, so the second element of these tuples is almost always 0).
#
# For example, below we define the module pool and task flow for the circle task:

# %%
import torch.nn as nn
from snorkel.classification.task import Operation

# Define a two-layer MLP module and a one-layer prediction "head" module
base_mlp = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
head_module = nn.Linear(4, 2)

# The module pool contains all the modules this task uses
module_pool = nn.ModuleDict({"base_mlp": base_mlp, "circle_head_module": head_module})

# "From the input dictionary, pull out 'circle_data' and send it through input_module"
op1 = Operation(
    name="input_op", module_name="base_mlp", inputs=[("_input_", "circle_data")]
)

# "From the output of op1 (the input op), pull out the 0th indexed output
# (i.e., the only output) and send it through the head_module"
op2 = Operation(
    name="head_op", module_name="circle_head_module", inputs=[("input_op", 0)]
)

task_flow = [op1, op2]

# %% [markdown]
# The output of the final module in that sequence will then go into a `loss_func()` to calculate the loss (e.g., cross-entropy) during training or an `output_func()` (e.g., softmax) to convert the logits into a prediction.  Each `Task` also specifies which metrics it supports, which are bundled together in a `Scorer` object. For this tutorial, we'll just look at accuracy.

# %% [markdown]
# Putting this all together, we define the circle task:

# %%
from functools import partial

from snorkel.classification.task import Task, ce_loss, softmax
from snorkel.classification.scorer import Scorer

circle_task = Task(
    name="circle_task",
    module_pool=module_pool,
    task_flow=task_flow,
    loss_func=partial(ce_loss, "head_op"),
    output_func=partial(softmax, "head_op"),
    scorer=Scorer(metrics=["accuracy"]),
)

# %% [markdown]
# Note that `Task` objects are not dependent on a particular dataset; multiple datasets can be passed through the same modules for pre-training or co-training.

# %% [markdown]
# We'll now define the square task, but more succinctly—for example, using the fact that the default name for an `Operation` is its `module_name` (since most tasks only use their modules once per forward pass). We'll also define the square task to share the input module with the circle task to demonstrate how to share modules. (Note that this is purely for illustrative purposes; for this toy task, it is very possible that this is not the optimal arrangement of modules).

# %%
square_task = Task(
    name="square_task",
    module_pool=nn.ModuleDict({"base_mlp": base_mlp, "square_head": nn.Linear(4, 2)}),
    task_flow=[
        Operation("base_mlp", [("_input_", "square_data")]),
        Operation("square_head", [("base_mlp", 0)]),
    ],
    loss_func=partial(ce_loss, "square_head"),
    output_func=partial(softmax, "square_head"),
    scorer=Scorer(metrics=["accuracy"]),
)

# %% [markdown]
# ## Model

# %% [markdown]
# With our tasks defined, constructing a model is simple: we simply pass the list of tasks in and the model constructs itself using information from the task flows.

# %%
from snorkel.classification.snorkel_classifier import SnorkelClassifier

model = SnorkelClassifier([circle_task, square_task])

# %% [markdown]
# ### Train Model

# %% [markdown]
# Once the model is constructed, we can train it as we would a single-task model, using the `train_model` method of a `Trainer` object. The `Trainer` supports multiple schedules or patterns for sampling from different dataloaders; the default is to randomly sample from them proportional to their number of batches, such that all examples  will be seen exactly once before any are seen twice.

# %%
from snorkel.classification.training import Trainer

trainer_config = {
    "progress_bar": True,
    "n_epochs": 10,
    "lr": 0.02,
    "log_manager_config": {"counter_unit": "epochs", "evaluation_freq": 2.0},
}

trainer = Trainer(**trainer_config)
trainer.train_model(model, dataloaders)

# %% [markdown]
# ### Evaluate model

# %% [markdown]
# After training, we can call the model.score() method to see the final performance of our trained model.

# %%
model.score(dataloaders)

# %% [markdown]
# Task-specific metrics are recorded in the form `task/dataset/split/metric` corresponding to the task the made the predictions, the dataset the predictions were made on, the split being evaluated, and the metric being calculated.
#
# For model-wide metrics (such as the total loss over all tasks or the learning rate), the default task name is `model` and the dataset name is `all` (e.g. `model/all/train/loss`).

# %% [markdown]
# ## Summary

# %% [markdown]
# In this tutorial, we demonstrated how to specify arbitrary flows through a network with  multiple datasets, providing the flexiblity to easily implement design patterns such as multi-task learning. On this toy task with only two simple datasets and very simple hard parameter sharing (a shared trunk with different heads), the utility of this design may be less apparent. However, for more complicated network structures (e.g., slicing) or scenarios with frequent changing of the structure (e.g., due to popping new tasks on/off a massive MTL model), the flexibility of this design starts to shine. If there's an MTL network you'd like to build but can't figure out how to represent, post an issue and let us know!
