# -*- coding: utf-8 -*-
# %% [markdown]
# # Multi-Task Learning (MTL) Basics Tutorial

# %% [markdown]
# Multi-task learning, or training a single model on multiple tasks, is becoming a standard tool for the modern ML practioner (see Ruder's [survey](http://ruder.io/multi-task/) from 2017 for a nice overview).
# It often leads to computational gains (one model performing many tasks takes up less memory and storage) as well as performance gains (learning to do well on a related _auxiliary_ task can improve the model's ability on the _primary_ task).
#
# While the primary purpose of the Snorkel project is to support training data creation and management, it also comes with a PyTorch-based modeling framework intended to support flexible multi-task learning (e.g. slice-aware models).
# Using this particular framework (as opposed to other excellent third party libraries) is entirely optional, but we have found it helpful in our own work and so provide it here.
# In particular, because MTL in general often requires easily *adding new datasets, tasks, and metrics* (and just as easily removing them), each of these concepts have been decoupled in the snorkel MTL classifier.

# %% [markdown]
# ### Tutorial Overview

# %% [markdown]
# The purpose of this tutorial is to introduce the basic interfaces and flow of the multi-task learning tools within Snorkel.
# We assume that you have prior experience with MTL, so we don't motivate or explain multi-task learning at large here.
#
# In this notebook, we will start by looking at a simple MTL model with only two tasks, each having distinct data and only one set of ground truth labels ("gold" labels). We'll also use a simple dataset where the raw data is directly usable as features, for simplicity (i.e., unlike text data, where we would first need to tokenize and transform the data into token ids).
# At the end, you'll fill in the missing details to add a third task to the model.

# %% [markdown]
# ## Environment Setup

# %%
# %matplotlib inline

from snorkel.utils import set_seed

SEED = 123
set_seed(SEED)

# %% [markdown]
# ## Create Toy Data

# %% [markdown]
# We'll now create a toy dataset to work with.
# Our data points are 2D points in a square centered on the origin.
# Our tasks will be classifying whether these points are:
#
# 1. Inside a **unit circle** centered on the origin (label 0 = `False`, label 1 = `True`)
# 2. Inside a **unit square** centered on the origin (label 0 = `False`, label 1 = `True`)
#
# We'll visualize these decision boundaries in a few cells.
#
# _Note: We don't expect these specific toy tasks to necessarily improve one another, but this is often a benefit of joint training in MTL settings when a model is trained on similar tasks._
# %%
import os

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("mtl")
# %%
from utils import make_circle_dataset, make_square_dataset

N = 1000  # Data points per dataset
R = 1  # Unit distance

circle_train, circle_valid, circle_test = make_circle_dataset(N, R)
(circle_X_train, circle_Y_train) = circle_train
(circle_X_valid, circle_Y_valid) = circle_valid
(circle_X_test, circle_Y_test) = circle_test

square_train, square_valid, square_test = make_square_dataset(N, R)
(square_X_train, square_Y_train) = square_train
(square_X_valid, square_Y_valid) = square_valid
(square_X_test, square_Y_test) = square_test

# %%
print(f"Training data shape: {circle_X_train.shape}")
print(f"Label space: {set(circle_Y_train)}")

# %% [markdown]
# And we can view the ground truth labels of our tasks visually to confirm our intuition on what the decision boundaries look like.
# In the plots below, the purple points represent class 0 and the yellow points represent class 1.

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)

scatter = axs[0].scatter(circle_X_train[:, 0], circle_X_train[:, 1], c=circle_Y_train)
axs[0].set_aspect("equal", "box")
axs[0].set_title("Circle Dataset", fontsize=10)
axs[0].legend(*scatter.legend_elements(), loc="upper right", title="Labels")

scatter = axs[1].scatter(square_X_train[:, 0], square_X_train[:, 1], c=square_Y_train)
axs[1].set_aspect("equal", "box")
axs[1].set_title("Square Dataset", fontsize=10)
axs[1].legend(*scatter.legend_elements(), loc="upper right", title="Labels")

plt.show()

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
from snorkel.classification import DictDataset, DictDataLoader

dataloaders = []
for (split, circle_X_split, circle_Y_split) in [
    ("train", circle_X_train, circle_Y_train),
    ("valid", circle_X_valid, circle_Y_valid),
    ("test", circle_X_test, circle_Y_test),
]:
    X_dict = {"circle_data": torch.FloatTensor(circle_X_split)}
    Y_dict = {"circle_task": torch.LongTensor(circle_Y_split)}
    dataset = DictDataset("CircleDataset", split, X_dict, Y_dict)
    dataloader = DictDataLoader(dataset, batch_size=32)
    dataloaders.append(dataloader)

for (split, square_X_split, square_Y_split) in [
    ("train", square_X_train, square_Y_train),
    ("valid", square_X_valid, square_Y_valid),
    ("test", square_X_test, square_Y_test),
]:
    X_dict = {"square_data": torch.FloatTensor(square_X_split)}
    Y_dict = {"square_task": torch.LongTensor(square_Y_split)}
    dataset = DictDataset("SquareDataset", split, X_dict, Y_dict)
    dataloader = DictDataLoader(dataset, batch_size=32)
    dataloaders.append(dataloader)

# %% [markdown]
# We now have 6 data loaders, one for each split (`train`, `valid`, `test`) of each task (`circle_task` and `square_task`).

# %% [markdown]
# ## Define Model

# %% [markdown]
# Now we'll define the `SnorkelClassifier` model, a PyTorch multi-task classifier.
# We'll instantiate it from a list of `Tasks`.

# %% [markdown]
# ### Tasks

# %% [markdown]
# A `Task` represents a path through a neural network. In `SnorkelClassifier`, this path corresponds to a particular sequence of PyTorch modules through which each example will make a forward pass.
#
# To specify this sequence of modules, each `Task` defines a **module pool** (a set of modules that it relies on) and a **task flow**—a sequence of `Operation`s.
# Each `Operation` specifies a module and the inputs to feed to that module.
# These inputs can come from a previous operation or the original input data.
# The inputs are defined by a list of tuples, where each tuple has the name of a previous operation (or the keyword `_input_` to denote the original input) and either the name of a field (e.g., if the output of that operation is a `dict`) or an index (e.g., if the output of that operation is a single `tensor`, a `list` or a `tuple`).
# Most PyTorch modules output a single tensor, so most of the time, the second element of this tuple is 0.
#
# As an example, below we verbosely define the module pool and task flow for the circle task:

# %%
import torch.nn as nn
from snorkel.classification import Operation

# Define a two-layer MLP module and a one-layer prediction "head" module
base_mlp = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
head_module = nn.Linear(4, 2)

# The module pool contains all the modules this task uses
module_pool = nn.ModuleDict({"base_mlp": base_mlp, "circle_head_module": head_module})

# "From the input dictionary, pull out 'circle_data' and send it through input_module"
op1 = Operation(
    name="base_mlp", module_name="base_mlp", inputs=[("_input_", "circle_data")]
)

# "From the output of op1 (the input op), pull out the 0th indexed output
# (i.e., the only output) and send it through the head_module"
op2 = Operation(
    name="circle_head", module_name="circle_head_module", inputs=[("base_mlp", 0)]
)

task_flow = [op1, op2]

# %% [markdown]
# A dictionary containing the outputs of all operations will then go into a `loss_func()` to calculate the loss (e.g., cross-entropy) during training or an `output_func()` (e.g., softmax) to convert the logits into a prediction.
# Both of these functions accept as an argument the name of the operation whose output they should use to calculate their respective values; in this case, that will be the `circle_head` operation.
# We indicate that here with the `partial` helper method, which can set the value of that keyword argument before the function is actually called.
# (As you'll see below, for common classification tasks, the default values for these arguments often suffice).
#
# Each `Task` also specifies which metrics it supports, which are bundled together in a `Scorer` object. For this tutorial, we'll just look at accuracy.

# %% [markdown]
# Putting this all together, we define the circle task:

# %%
from functools import partial

from snorkel.classification import Scorer, Task, ce_loss, softmax

circle_task = Task(
    name="circle_task",
    module_pool=module_pool,
    task_flow=task_flow,
    loss_func=partial(ce_loss, op_name="circle_head"),
    output_func=partial(softmax, op_name="circle_head"),
    scorer=Scorer(metrics=["accuracy"]),
)

# %% [markdown]
# Note that `Task` objects are not dependent on a particular dataset; multiple datasets can be passed through the same modules for pre-training or co-training.

# %% [markdown]
# ### Again, but faster

# %% [markdown]
# We'll now define the square task, but more succinctly—for example, using the fact that the default name for an `Operation` is its `module_name` (since most tasks only use their modules once per forward pass).
#
# We'll also define the square task to share the first module in its task flow (`base_mlp`) with the circle task to demonstrate how to share modules. (Note that this is purely for illustrative purposes; for this toy task, it is very possible that this is not the optimal arrangement of modules).
#
# Finally, the most common task definitions we see in practice are classification tasks with cross-entropy loss and softmax on the output of the last module, and accuracy is most often the primary metric of interest, these are all the default values, so we can drop them here for brevity.

# %%
square_task = Task(
    name="square_task",
    module_pool=nn.ModuleDict({"base_mlp": base_mlp, "square_head": nn.Linear(4, 2)}),
    task_flow=[
        Operation("base_mlp", [("_input_", "square_data")]),
        Operation("square_head", [("base_mlp", 0)]),
    ],
)

# %% [markdown]
# ## Model

# %% [markdown]
# With our tasks defined, constructing a model is simple: we simply pass the list of tasks in and the model constructs itself using information from the task flows.
#
# Note that the model uses the names of modules (not the modules themselves) to determine whether two modules specified by separate tasks are the same module (and should share weights) or different modules (with separate weights).
# So because both the `square_task` and `circle_task` include "base_mlp" in their module pools, this module will be shared between the two tasks.

# %%
from snorkel.classification import SnorkelClassifier

model = SnorkelClassifier([circle_task, square_task])

# %% [markdown]
# ### Train Model

# %% [markdown]
# Once the model is constructed, we can train it as we would a single-task model, using the `fit` method of a `Trainer` object. The `Trainer` supports multiple schedules or patterns for sampling from different dataloaders; the default is to randomly sample from them proportional to their number of batches, such that all examples  will be seen exactly once before any are seen twice.

# %%
from snorkel.classification import Trainer

trainer_config = {"progress_bar": False, "n_epochs": 10, "lr": 0.02}

trainer = Trainer(**trainer_config)
trainer.fit(model, dataloaders)

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
# # Your Turn

# %% [markdown]
# To check your understanding of how to use the multi-task `SnorkelClassifier`, see if you can add a task to this multi-task model.
#
# We'll generate the data for you (again, with a train, valid, and test split).
# Let's call it the `inv_circle_task`, since it will have the same distribution as our circle data, but with the inverted (flipped) labels.
# Intuitively, a model that is very good at telling whether a point is within a certain region should also be very good at telling if it's outside the region.
#
# By sharing some layers (the `base_mlp`), this new task will help the model to learn a representation that benefits the `circle_task` as well.
# And because it will have a non-shared layer (call it the `inv_circle_head`), it will have the flexibility to map that good representation into the right label space for its own task.

# %% [markdown]
# ### Create the data

# %%
from utils import make_inv_circle_dataset

# We flip the inequality when generating the labels so that our positive
# class is now _outside_ the circle.
inv_circle_train, inv_circle_valid, inv_circle_test = make_inv_circle_dataset(N, R)
(inv_circle_X_train, inv_circle_Y_train) = inv_circle_train
(inv_circle_X_valid, inv_circle_Y_valid) = inv_circle_valid
(inv_circle_X_test, inv_circle_Y_test) = inv_circle_test

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3)

scatter = axs[0].scatter(
    inv_circle_X_train[:, 0], inv_circle_X_train[:, 1], c=inv_circle_Y_train
)
axs[0].set_aspect("equal", "box")
axs[0].set_title("Inv Circle Dataset", fontsize=10)
axs[0].legend(*scatter.legend_elements(), loc="upper right", title="Labels")

scatter = axs[1].scatter(circle_X_train[:, 0], circle_X_train[:, 1], c=circle_Y_train)
axs[1].set_aspect("equal", "box")
axs[1].set_title("Circle Dataset", fontsize=10)
axs[1].legend(*scatter.legend_elements(), loc="upper right", title="Labels")

scatter = axs[2].scatter(square_X_train[:, 0], square_X_train[:, 1], c=square_Y_train)
axs[2].set_aspect("equal", "box")
axs[2].set_title("Square Dataset", fontsize=10)
axs[2].legend(*scatter.legend_elements(), loc="upper right", title="Labels")


plt.show()

# %% [markdown]
# ### Create the DictDataLoader

# %% [markdown]
# Create the `DictDataLoader` for this new dataset.
# - The X_dict should map data field names to data (in this case, we only need one field, since our data is represented by a single Tensor). You can name the field whatever you want; you'll just need to make sure that your `Task` object refers to the right field name in its task flow.
# - The Y_dict should map a task name to a set of labels. This will tell the model what path through the network to use when making predictions or calculating loss on batches from this dataset. At this point we haven't yet defined our

# %%
X_dict = {}  # Filled in by you
Y_dict = {}  # Filled in by you
inv_dataset = DictDataset("InvCircleDataset", "train", X_dict, Y_dict)
inv_dataloader = DictDataLoader(dataset=inv_dataset, batch_size=32)

# %% [markdown]
# We add this new dataloader to the dataloaders for the other tasks.

# %%
all_dataloaders = dataloaders + [inv_dataloader]

# %% [markdown]
# ### Create the task

# %% [markdown]
# Using the `square_task` definition as a template, fill in the arguments for an `inverse_circle_task` that consists of the same `base_mlp` module as the other tasks and a separate linear head with an output of size 2.

# %%
# Uncomment and fill in the arguments to create a Task object for the inverse_circle task.
# inv_circle_task = Task(
#     name="",  # Filled in by you
#     module_pool=nn.ModuleDict({}),  # Filled in by you
#     task_flow=[],  # Filled in by you
# )

# %% [markdown]
# ### Create the model

# %% [markdown]
# Once we have our task objects, creating the new multi-task model is as easy as adding the new task to the list of tasks at model initialization time.

# %%
# Add your new task to the list of tasks for creating the MTL model
model = SnorkelClassifier([circle_task, square_task])  # Filled in by you

# %% [markdown]
# ### Train the model

# %% [markdown]
# We can use the same trainer and training settings as before.

# %%
trainer.fit(model, all_dataloaders)
model.score(all_dataloaders)

# %% [markdown]
# ### Validation

# %% [markdown]
# If you successfully added the appropriate task, the previous command should have succesfully trained and reported scores in the mid to high 90s for all datasets and splits, including for the splits belonging to the new `inv_circle_task`.
# The following assert statements should also pass if you uncomment and run it.

# %%
# assert len(model.tasks) == 3
# assert len(model.module_pool) == 4  # 1 shared module plus 3 separate task heads

# %% [markdown]
# ## Summary

# %% [markdown]
# In this tutorial, we demonstrated how to specify arbitrary flows through a network with  multiple datasets, providing the flexiblity to easily implement design patterns such as multi-task learning. On this toy task with only two simple datasets and very simple hard parameter sharing (a shared trunk with different heads), the utility of this design may be less apparent. However, for more complicated network structures (e.g., slicing) or scenarios with frequent changing of the structure (e.g., due to popping new tasks on/off a massive MTL model), the flexibility of this design starts to shine. If there's an MTL network you'd like to build but can't figure out how to represent, post an issue and let us know!
