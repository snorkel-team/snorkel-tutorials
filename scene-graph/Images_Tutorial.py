# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
# # Generating Weak Labels for Image Datasets (e.g. `Person Riding Bike`)

# %% [markdown]
# _Note_: This notebook assumes that Snorkel is installed. If not, see the [Quick Start guide](https://github.com/HazyResearch/snorkel#quick-start) in the Snorkel README.
#
# ---
# In this tutorial, we write labeling functions over a set of unlabeled images to create a weakly-labeled dataset for **person riding bike**. 
#
# 1. **Load and visualize dataset** — Build intuition about heuristics and weak supervision for our task.
# 2. **Generate Primitives** — Writing labeling functions over raw image pixels is quite difficult for most tasks. Instead, we first create low level primitives (e.g. bounding box sizes/positions), which we can then write LFs over.
# 3. **Write Labeling Functions** — Express our heuristics as labeling functions over user-defined primitives.
# 4. **Generate Training Set** — Aggregate our heuristic-based lableing functions to create a training set using the Snorkel paradigm.
#
# This process can be viewed as a way of leveraging off-the-shelf tools we already have (e.g. pretrained models, object detectors, etc.) for new tasks, which is very similar to the way we leveraged text processing tools in the other tutorials!
#
# In this approach, we show that incorporating Snorkel's generative model for labeling function aggregation shows a significant lift in accuracy over majority vote. 
#
# While the tutorial we show takes advantage of very basic primitives and models, there's a lot of room to experiment here. For more, see recent work ([Varma et. al 2017](https://arxiv.org/abs/1709.02477)) that incorporates static analysis + primitive dependencies to infer structure in generative models.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import os

# %% [markdown]
# _Note_: In this tutorial, we use `scikit-image`, which isn't a `snorkel` dependency. If you don't already have it installed, please run the following cell.

# %% [markdown]
# ## 1. Load and Visualize Dataset
# First, we load the dataset and associated bounding box objects and labels for people riding bikes.

# %%
from data_loader import DataLoader
loader = DataLoader()

# %% [markdown]
# We can visualize some **positive examples** of people riding bikes...

# %%
loader.show_examples(annotated=False, label=1)

# %% [markdown]
# ...and some **negative** examples.

# %%
loader.show_examples(annotated=False, label=-1)


# %% [markdown]
# ## 2. Generate Primitives
# We write labeling functions (LFs) over _primitives_, instead of raw pixel values, becaues they are practical and interpretable.
#
# For this dataset, each image comes with object bounding boxes, extracted using off-the-shelf tools. We use the labels, positions, and sizes of each object as simple _primitives_ and also combine them into more complex _primitives_, as seen in `primitive_helpers.py`.
#
# _Note_: For this tutorial, we generate very simple primitives using off-the-shelf methods, but there is a lot of room for development and exploration here!

# %% [markdown]
# #### Membership-based Primitives
# These _primitives_ check whether certain objects appear in the images.

# %%
def has_bike(object_names):
    if ('cycle' in object_names) or ('bike' in object_names) or ('bicycle' in object_names):
        return 1
    else:
        return 0


# %%
def has_human(object_names):
    if (('person' in object_names) or ('woman' in object_names) or ('man' in object_names)) \
        and (('bicycle' in object_names) or 'bicycles' in object_names):
        return 1
    else:
        return 0


# %%
def has_road(object_names):
    if ('road' in object_names) or ('street' in object_names) or ('concrete' in object_names):
        return 1
    else:
        return 0


# %%
def has_cars(object_names):
    if ('car' in object_names) or ('cars' in object_names) or \
        ('bus' in object_names) or ('buses' in object_names) or \
        ('truck' in object_names) or ('trucks' in object_names):
        return 1
    else:
        return 0


# %% [markdown]
# #### Object Relationship Based Primitives
# These _primitives_ look at the relations among bikes and people in the images. They capture the relative...
# * position of bikes vs people (via `bike_human_distance`)
# * number of bikes vs people (via `bike_human_nums`)
# * size of bikes vs people (via `bike_human_size`)
#
# The code for these _primitives_ can be found `primitive_helpers.py`.

# %%
from primitive_helpers import bike_human_distance, bike_human_size, bike_human_nums

def create_primitives(loader):
    m = 7 # number of primitives
    primitive_mtx = np.zeros((loader.train_num,m))

    for i in range(loader.train_num):
        primitive_mtx[i,0] = has_human(loader.train_object_names[i])
        primitive_mtx[i,1] = has_road(loader.train_object_names[i])
        primitive_mtx[i,2] = has_cars(loader.train_object_names[i])
        primitive_mtx[i,3] = has_bike(loader.train_object_names[i])

        primitive_mtx[i,4] = bike_human_distance(loader.train_object_names[i], 
                                                 loader.train_object_x[i], 
                                                 loader.train_object_y[i])

        area = np.multiply(loader.train_object_height[i], loader.train_object_width[i])
        primitive_mtx[i,5] = bike_human_size(loader.train_object_names[i], area)
        primitive_mtx[i,6] = bike_human_nums(loader.train_object_names[i])

    return primitive_mtx


# %% [markdown]
# **Assign and Name Primitives**
#
# We assign the primitives and name them according to the variables we will use to refer to them in the labeling functions we develop next. For example, `primitive_mtx[:,0]` is referred to as `has_human`.

# %%
primitive_mtx = create_primitives(loader)

p_keys = {
    'has_human': primitive_mtx[:,0],
    'has_road': primitive_mtx[:, 1],
    'has_cars': primitive_mtx[:, 2],
    'has_bike': primitive_mtx[:, 3],
    'bike_human_distance': primitive_mtx[:, 4],
    'bike_human_size': primitive_mtx[:, 5],
    'bike_human_num': primitive_mtx[:, 6]
   }


# %% [markdown]
# ## 3. Write Labeling Functions (LFs)
# We now develop LFs that take different primitives in as inputs and apply a label based on the value of those primitives. Notice that each of these LFs are "weak"— they aren't fully precise, and they don't have complete coverage.
#
# Below, we have incldue the intuition that explains each of the LFs: 
# * `LF_street`: If the image has a human and a road, we think a person might be riding a bike. 
# * `LF_vechicles`: If the image has a human and a vehicle, we think a person might be riding a bike.
# * `LF_distance`: If the image has a human and bike close to one another, we think that a person might be riding a bike. 
# * `LF_size`: If the image has a human/bike around the same size (perhaps they're both in the foreground or background), we think a person might be riding a bike. 
# * `LF_number`: If the image has the same number of bicycles and humans (i.e. _primitive_ categorized as `bike_human_num=2`) or there are fewer humans than bikes (i.e. `bike_human_num=0`), we think a person might be riding a bike.

# %%
def LF_street(has_human, has_road):
    if has_human >= 1: 
        if has_road >= 1:
            return 1
        else:
            return -1
    return 0

def LF_vehicles(has_human, has_cars):
    if has_human >= 1: 
        if has_cars >= 1:
            return 1
        else:
            return -1
    return 0

def LF_distance(has_human, has_bike, bike_human_distance):
    if has_human >= 1:
        if has_bike >= 1: 
            if bike_human_distance <= np.sqrt(8):
                return 1
            else:
                return 0
    else:
        return -1
    
def LF_size(has_human, has_bike, bike_human_size):
    if has_human >= 1:
        if has_bike >= 1: 
            if bike_human_size <= 1000:
                return -1
            else:
                return 0
    else:
        return -1
    
    
def LF_number(has_human, has_bike, bike_human_num):
    if has_human >= 1:
        if has_bike >= 1: 
            if bike_human_num >= 2:
                return 1
            if bike_human_num >= 1:
                return 0
            if bike_human_num >= 0:
                return 1
    else:
        return -1


# %% [markdown]
# **Assign Labeling Functions**
#
# We create a list of the functions we used in `L_fns` and apply the labeling functions to the appropriate primitives to generate `L`, a _labeling matrix_.
#
# _Note_: We usually have Snorkel manage our data using its ORM database backend, in which case we use the `LabelAnnotator` from the `snorkel.annotations` [module](http://snorkel.readthedocs.io/en/master/annotations.html#snorkel.annotations.LabelAnnotator). In this tutorial, we show how to explicitly construct labeling matrices manually, which can be useful when managing your data outside of Snorkel, as is the case with our image data!

# %%
L_fns = [LF_street,LF_vehicles,LF_distance,LF_size,LF_number]

L = np.zeros((len(L_fns),loader.train_num)).astype(int)
for i in range(loader.train_num):
    L[0,i] = L_fns[0](p_keys['has_human'][i], p_keys['has_road'][i])
    L[1,i] = L_fns[1](p_keys['has_human'][i], p_keys['has_cars'][i])
    L[2,i] = L_fns[2](p_keys['has_human'][i], p_keys['has_bike'][i], p_keys['bike_human_distance'][i])
    L[3,i] = L_fns[3](p_keys['has_human'][i], p_keys['has_bike'][i], p_keys['bike_human_size'][i])
    L[4,i] = L_fns[4](p_keys['has_human'][i], p_keys['has_bike'][i], p_keys['bike_human_num'][i])

# %% [markdown]
# **Calculate and Show Accuracy and Coverage of Labeling Functions**
#
# Notice that while the labeling functions were intuitive for humans to write, they do not perform particularly well on their own. _Hint_: this is where the magic of Snorkel's generative model comes in!
#
# _Note_: we define _coverage_ as the proportion of samples from which an LF does not abstain. Recall that each "uncertain" labeling function assigns `0`.

# %%
total = float(loader.train_num)

stats_table = np.zeros((len(L),2))
for i in range(len(L)):
    # coverage: (num labeled) / (total)
    stats_table[i,0] = np.sum(L[i,:] != 0)/ total
    
    # accuracy: (num correct assigned labels) / (total assigned labels)
    stats_table[i,1] = np.sum(L[i,:] == loader.train_ground)/float(np.sum(L[i,:] != 0))

# %%
import pandas as pd
stats_table = pd.DataFrame(stats_table, index = [lf.__name__ for lf in L_fns], columns = ["Coverage", "Accuracy"])
stats_table

# %% [markdown]
# ## 4. Generate Training Set
# At this point, we can take advantage of Snorkel's generative model to aggregate labels from our noisy labeling functions. 

# %%
from snorkel.learning import GenerativeModel
from scipy import sparse
import matplotlib.pyplot as plt

L_train = sparse.csr_matrix(L.T)

# %% [markdown]
# **Majority Vote**
#
# To get a sense of how well our labeling functions perform when aggregated, we calcuate the accuracy of the training set labels if we took the majority vote label for each data point. This gives us a baseline for comparison against Snorkel's generative model.

# %%
mv_labels = np.sign(np.sum(L.T,1))
print ('Coverage of Majority Vote on Train Set: ', np.sum(np.sign(np.sum(np.abs(L.T),1)) != 0)/float(loader.train_num))
print ('Accuracy of Majority Vote on Train Set: ', np.sum(mv_labels == loader.train_ground)/float(loader.train_num))

# %% [markdown]
# **Generative Model**
# For the Snorkel generative model, we assume that the labeling functions are conditionally independent given the true label. We train the generative model using the labels assigned by the labeling functions. 
#
# For more advanced modeling of generative structure (i.e. using dependencies between primitives), refer to the Coral paradigm, as described in [Varma et. al 2017](https://arxiv.org/abs/1709.02477).

# %%
gen_model = GenerativeModel()

gen_model.train(L.T, epochs=100, decay=0.95, step_size= 0.01/ L.shape[1], reg_param=1e-6)
train_marginals = gen_model.marginals(L_train)

# %% [markdown]
# **Probabilistic Label Statistics**
#
# We view the distribution of weak labels produced by our generative model. 

# %%
plt.hist(train_marginals, bins=20)
plt.show()

# %% [markdown]
# We can also compare the **empirical** accuracies of our labeling functions to the **learned** accuracies of our generative model over the validation data. 

# %%
learned_table = gen_model.learned_lf_stats()
empirical_acc = stats_table.values[:, 1]
learned_acc = learned_table.values[:,0]
compared_stats = pd.DataFrame(np.stack((empirical_acc, learned_acc)).T,
                             index = [lf.__name__ for lf in L_fns],
                             columns=['Empirical Acc.', 'Learned Acc.'])

compared_stats

# %% [markdown]
# _Note_: Coverage still refers to our model's tendency abstention from assigning labels to examples in the dataset. In this case, Snorkel's generative model has full coverage as it generalizes the labeling functions— it never assigns the "uncertain" label of 0.5.

# %%
labels = 2 * (train_marginals > 0.9) - 1
print ('Coverage of Generative Model on Train Set:', np.sum(train_marginals != 0.5)/float(len(train_marginals)))
print ('Accuracy of Generative Model on Train Set:', np.mean(labels == loader.train_ground))

# %% [markdown]
# Take note that with Snorkel's generative model, we're able to achieve a much higher accuracy than with majority vote over the labeling functions!

# %% [markdown]
# ## Now, train a discriminative model with your weak labels! 
# You can then use these training labels to train any standard discriminative model, such as [a state-of-the-art ResNet](https://github.com/KaimingHe/deep-residual-networks), which should learn to generalize beyond the LF's we've developed!
#
# The only change needed from standard procedure is to deal with the fact that the training labels Snorkel generates are _probabilistic_ (i.e. for the binary case, in [0,1])— luckily, this is a one-liner in most modern ML frameworks! For example, in TensorFlow, you can use the [cross-entropy loss](https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits).

# %%
