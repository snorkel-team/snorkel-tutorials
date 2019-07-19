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
# # Classifying Chest X-rays with Cross-Modal Data Programming

# %% [markdown]
# This tutorial demonstrates how to use the *cross-modal data programming* technique described in Dunnmon and Ratner, et al. (2019) to build a Convolutional Neural Network (CNN) model with no hand-labeled data that performs similarly to a CNN supervised using several thousand data points labeled by radiologists.  This process is *exactly* equivalent to that followed for the chest radiograph dataset in our 2019 Nature submission.
#
# We begin by setting up our environment, importing relevant Python packages.

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# Importing pandas for data processing
import pandas as pd

# %% [markdown]
# ## Step 1: Loading and Splitting the Data

# %% [markdown]
# First, we set up the data dictionary and load data that we've already split for you into an (approximately) 80% train split, 10% development split, and 10% test split.  Each raw data point contains three fields: a text report, a label (normal or abnormal), and a set of image paths.  The original data, from the OpenI dataset, is maintained by [NIH](https://openi.nlm.nih.gov/faq.php).

# %% [markdown]
# **TODO:**
# 1. Move the dataset adjustment and conversion to a preprocessing script
# 2. Do not include labels column in train

# %%
# Setting up data dictionary and defining data splits
data = {}
splits = ["train", "dev", "test"]

for split in splits:
    data[split] = pd.read_csv(f"data/{split}_entries.csv")[
        ["label", "xray_paths", "text"]
    ]
    # Adjusting labels to fit with Snorkel MeTaL labeling convention
    data[split]["label"][data[split]["label"] == 0] = 2
    perc_pos = sum(data[split]["label"] == 1) / len(data[split])
    print(f"{len(data[split])} {split} examples: {100*perc_pos:0.1f}% Abnormal")
    
train_df = pd.DataFrame.from_dict(data["train"])
valid_df = pd.DataFrame.from_dict(data["dev"])
test_df = pd.DataFrame.from_dict(data["test"])

valid_df.head()

# %% [markdown]
# You can see an example of a single data point below -- note that the raw label convention for our normal vs. abnormal classification problem is 1 for abnormal and 2 for normal.

# %%
sample = valid_df.iloc[0]
print("RAW TEXT:\n \n", sample["text"], "\n")
print("IMAGE PATHS: \n \n", sample["xray_paths"], "\n")
print("LABEL:", sample["label"])

# %% [markdown]
# ## Step 2: Developing LFs

# %% [markdown]
# We now define our *labeling functions* (LFs): simple, heuristic functions written by a domain expert (e.g., a radiologist) that correctly label a report as normal or abnormal with probability better than random chance.
#
# We give an example of all three types of LFs we reference in the paper: general pattern LFs that operate on patterns a non-expert user could easily identify, medical pattern LFs that operate on patterns easily identifiable by a clinician, and structural LFs that focus on specific structural elements of the report (e.g. how long it is) that have some correlation with the scan it describes being normal or abnormal.
#
# **TODO:**
# 1. Convert regex things to preprocessors

# %%
from snorkel.labeling.apply import PandasLFApplier
from snorkel.labeling.lf import labeling_function
import re

# Value to use for abstain votes
ABSTAIN = 0
# Value to use for abnormal votes
ABNORMAL = 1
# Value to user for normal votes
NORMAL = 2

# %%
lfs = []

# Example of a General Pattern LF
noted_or_seen = ["is seen", "noted"]
@labeling_function(resources=dict(noted_or_seen=noted_or_seen))
def LF_noted_or_seen(x, noted_or_seen):
    if any(word in x.text.lower() for word in noted_or_seen):
        return ABNORMAL
    else:
        return ABSTAIN
lfs.append(LF_noted_or_seen)

negative_words = ["but", "however", "otherwise"]
@labeling_function(resources=dict(negative_words=negative_words))
def LF_negative(x, negative_words):
    return (
        ABNORMAL
        if any(word in x.text.lower() for word in negative_words)
        else ABSTAIN
    )
lfs.append(LF_negative)

@labeling_function()
def LF_disease_in_report(x):
    return ABNORMAL if "disease" in x.text.lower() else ABSTAIN
lfs.append(LF_disease_in_report)

@labeling_function()
def LF_recommend_in_report(x):
    return ABNORMAL if "recommend" in x.text.lower() else ABSTAIN
lfs.append(LF_recommend_in_report)

@labeling_function()
def LF_mm_in_report(x):
    return ABNORMAL if any(word in x.text.lower() for word in ["mm", "cm"]) else ABSTAIN
lfs.append(LF_mm_in_report)

# Example of a Medical Pattern LF
@labeling_function()
def LF_lung_hyperdistention_demo(x):
    """
    Votes abnormal for indications of lung hyperdistention.
    """
    reg_01 = re.compile("increased volume|hyperexpan|inflated", re.IGNORECASE)
    for s in x.text.split("."):
        if reg_01.search(s):
            return ABNORMAL
    return ABSTAIN
lfs.append(LF_lung_hyperdistention_demo)

normalcy_words = ["clear", "no", "normal", "unremarkable", "free", "midline"]
@labeling_function(resources=dict(normalcy_words=normalcy_words))
def LF_consistency_in_report(x, normalcy_words):
    """
    The words 'clear', 'no', 'normal', 'free', 'midline' in
    findings section of the report
    """
    findings = x.text[x.text.find("FINDINGS:") :]
    findings = findings[: findings.find("IMPRESSION:")]
    sents = findings.split(".")

    num_sents_without_normal = ABSTAIN
    for sent in sents:
        sent = sent.lower()
        if not any(word in sent for word in normalcy_words):
            num_sents_without_normal += 1
        elif "not" in sent:
            num_sents_without_normal += 1
    return NORMAL if num_sents_without_normal < 2 else ABNORMAL
lfs.append(LF_consistency_in_report)

abnormal_mesh_terms = [
    "opacity",
    "cardiomegaly",
    "calcinosis",
    "hypoinflation",
    "calcified granuloma",
    "thoracic vertebrae",
    "degenerative",
    "hyperdistention",
    "catheters",
    "granulomatous",
    "nodule",
    "fracture" "surgical",
    "instruments",
    "emphysema",
]
@labeling_function(resources=dict(abnormal_mesh_terms=abnormal_mesh_terms))
def LF_abnormal_mesh_terms_in_report(x, abnormal_mesh_terms):
    if any(mesh in x.text.lower() for mesh in abnormal_mesh_terms):
        return ABNORMAL
    else:
        return ABSTAIN
lfs.append(LF_abnormal_mesh_terms_in_report)

# Example of a Structural LF
@labeling_function()
def LF_report_is_short_demo(x):
    """
    Checks if report is short.
    """
    return NORMAL if len(x.text) < 280 else ABSTAIN
lfs.append(LF_report_is_short_demo)

# %% [markdown]
# Now, we can see how well these LFs might do at correctly indicating normal or abnormal examples by first applying the labeling functions to the examples and then printing some useful statistics.

# %%
from snorkel.labeling.apply import PandasLFApplier

applier = PandasLFApplier(lfs)
L_train = applier.apply(train_df)
L_valid = applier.apply(valid_df)

# %% [markdown]
# If we use analyze the `LF_lung_hyperdistention_demo` function -- in this case,  we see that it has polarity [1,2], meaning it votes on both class 1 and class 2 (and votes on every example because `coverage` = 1.0), but that it has low accuracy (around 44%).  Let's look at the confusion matrix to see why.

# %%
from snorkel.analysis.utils import convert_labels
from snorkel.labeling.analysis import lf_summary

Y_valid = valid_df.label.values
lf_names= [lf.name for lf in lfs]
lf_summary(L_valid, Y_valid, lf_names=lf_names)

# %% [markdown]
# **TODO: don't have interactive components that require users to change for good result in the tutorial**
#
# Clearly, this LF is much more accurate on abnormal examples (where y=1) than on abnormal examples (where y=2).  Why don't we adjust it to only vote in the positive direction and see how we do?
#
# Go ahead and change `NORMAL` to `ABSTAIN` in the `LF_lung_hyperdistention_demo` function (the line below the `### *** ###` comment), and rerun the last three code cells.
#
# You'll see that by making this rule a bit more targeted, its coverage decreases to 9%, but it's accuracy jumps to over 90%.  This type of iteration is exactly how clinicians can develop LFs in practice.
#
# You may also notice that it's very easy to write these LFs over text, but it would be very hard to, say, write an `LF_lung_hyperdistention` version that operates over an image -- this is why cross-modality is so important!

# %% [markdown]
# Once we've designed a couple of LFs, it's time to execute them all on every example we have to create a *label matrix*.  This is an $n$ by $m$ matrix, where $n$ is the number of examples and $m$ is the number of LFs.

# %%
# from labeling_functions import (
#     LF_report_is_short,
#     LF_consistency_in_report,
#     LF_negative_inflection_words_in_report,
#     LF_is_seen_or_noted_in_report,
#     LF_disease_in_report,
#     LF_abnormal_mesh_terms_in_report,
#     LF_recommend_in_report,
#     LF_mm_in_report,
#     LF_normal,
#     LF_positive_MeshTerm,
#     LF_fracture,
#     LF_calcinosis,
#     LF_degen_spine,
#     LF_lung_hypoinflation,
#     LF_lung_hyperdistention,
#     LF_catheters,
#     LF_surgical,
#     LF_granuloma,
# )

# lfs = [
#     LF_report_is_short,
#     LF_consistency_in_report,
#     LF_negative_inflection_words_in_report,
#     LF_is_seen_or_noted_in_report,
#     LF_disease_in_report,
#     LF_abnormal_mesh_terms_in_report,
#     LF_recommend_in_report,
#     LF_mm_in_report,
#     LF_normal,
#     LF_positive_MeshTerm,
#     LF_fracture,
#     LF_calcinosis,
#     LF_degen_spine,
#     LF_lung_hypoinflation,
#     LF_lung_hyperdistention,
#     LF_catheters,
#     LF_surgical,
#     LF_granuloma,
# ]

# %% [markdown]
# ## Step 4: Train a Label Model in Snorkel MeTaL

# %% [markdown]
# Next, we use the Snorkel MeTaL model training API to train a `LabelModel` that learns the accuracies of our LFs.  This is the core step that the data programming technique simplifies and formalizes -- by combining our labeling functions based on their accuracies, we can recover a model that outputs reasonable weak labels.
#
# We perform a simple random hyperparameter search over learning rate and L2 regularization, using our small labeled development set to choose the best model.

# %%
from snorkel.labeling.model import LabelModel
from snorkel.analysis.utils import probs_to_preds
from snorkel.analysis.metrics import metric_score

label_model = LabelModel(cardinality=2, verbose=True)
label_model.train_model(L_train, log_train_every=10, lr=0.05, class_balance=[0.7, 0.3], n_epochs=100)

# %%
Y_probs_valid = label_model.predict_proba(L_valid)
Y_preds_valid = probs_to_preds(Y_probs_valid)
metric_score(Y_valid, Y_preds_valid, probs=None, metric="f1")

# %% [markdown]
# **Majority Vote**

# %%
from snorkel.labeling.model import MajorityLabelVoter

mv_model = MajorityLabelVoter()
Y_probs_valid = mv_model.predict_proba(L_valid)
Y_preds_valid = probs_to_preds(Y_probs_valid)
metric_score(Y_valid, Y_preds_valid, probs=None, metric="f1")

# %%
# from metal.tuners import RandomSearchTuner

# # Creating search space
# search_space = {
#     "l2": {"range": [0.0001, 0.1], "scale": "log"},  # linear range
#     "lr": {"range": [0.0001, 0.1], "scale": "log"},  # log range
# }

# searcher = RandomSearchTuner(LabelModel, log_dir="./run_logs", log_writer_class=None)

# # Training label model


# label_model = searcher.search(
#     search_space,
#     (Ls[1], Ys[1]),
#     train_args=[Ls[0]],
#     init_args=[],
#     init_kwargs={"k": 2, "seed": 1701},
#     train_kwargs={"n_epochs": 200},
#     max_search=40,
#     verbose=True,
# )

# %% [markdown]
# We evaluate our best model on the development set as below -- you should recover a model with best accuracy of approximately 85% on the development set -- this `LabelModel`will be applied to the training set to create weak labels, which we can then use to train our image classifier.

# %%
# BLOCKED

# Getting scores
# scores = label_model.score(
#     (Ls[1], Ys[1]), metric=["accuracy", "precision", "recall", "f1"]
# )

# %% [markdown]
# Why is this useful?  If we compare to majority vote, we see a couple points of improvement in accuracy.  Note that the degree to which we expect this model to improve over majority vote varies based on the type of dataset involved, as detailed in the 2018 [VLDB Paper](http://www.vldb.org/pvldb/vol11/p269-ratner.pdf) describing the Snorkel system.

# %%
# BLOCKED

# from snorkel.labeling.model.baselines import MajorityLabelVoter

# Checking if we beat majority vote
# # mv = MajorityLabelVoter(seed=123)
# scores = mv.score((Ls[1], Ys[1]), metric=["accuracy", "precision", "recall", "f1"])

# %% [markdown]
# ## Step 5: Create a Weakly Labeled Training Set

# %% [markdown]
# We can now use this trained `LabelModel` to create weak labels for each of our train, development, and test splits by applying it to the label matrices, as below.

# %%
Y_train_ps = label_model.predict_proba(Ls[0])
Y_dev_ps = label_model.predict_proba(Ls[1])
Y_test_ps = label_model.predict_proba(Ls[2])
Y_ps = [Y_train_ps, Y_dev_ps, Y_test_ps]

# %% [markdown]
# We can inspect the distribution of our weak training labels, and note that they are assigned varying degrees of probability.  An advantage of this labeling approach is that probabilistic labels can be very descriptive -- i.e., if an example has a 60% probability of being abnormal, we train against that 0.6 probability, rather than binarizing to 100%.

# %%
# TODO

# from metal.contrib.visualization.analysis import plot_probabilities_histogram

# Looking at probability histogram for training labels
# plot_probabilities_histogram(Y_dev_ps[:, 0], title="Probablistic Label Distribution")

# %% [markdown]
# Using the development set, we can also check that the class balance of our weak labels if we were to naively binarize at the 0.5 cutoff -- we see reasonable behavior here.

# %%
# TODO

# from metal.contrib.visualization.analysis import plot_predictions_histogram

# Obtaining binarized predictions
# Y_dev_p = label_model.predict(Ls[1])
# plot_predictions_histogram(Y_dev_p, Ys[1], title="Label Distribution")

# %% [markdown]
# ## Step 6: Train a Weakly Supervised End Model

# %% [markdown]
# Now that we have our weak training labels, we can train a commodity CNN using a simple PyTorch API.  In Snorkel MeTaL, we have written high-level utilities to do this.  The entire process of defining and training the model can be executed in the following two simple cells.
#
# First, we define PyTorch `DataLoader` objects to efficiently load our image data, associating each image with the weak label generated from its associated report.

# %%
# TODO: Replace end model

# import torch
# from torchvision import models
# from metal.end_model import EndModel
# from metal.logging.tensorboard import TensorBoardWriter
# from utils import get_data_loader

# # Setting up log directory
# log_config = {"log_dir": "./run_logs", "run_name": "openi_demo_ws"}
# tuner_config = {"max_search": 1}
# search_space = {"l2": [0.0005], "lr": [0.001]}  # linear range

# # Create pytorch model
# num_classes = 2
# cnn_model = models.resnet18(pretrained=True)
# last_layer_input_size = int(cnn_model.fc.weight.size()[1])
# cnn_model.fc = torch.nn.Linear(last_layer_input_size, num_classes)

# # Create data loaders
# loaders = {}
# loaders["train"] = get_data_loader(
#     data["train"]["xray_paths"].tolist(), Y_ps[0], batch_size=32, shuffle=True
# )
# loaders["dev"] = get_data_loader(
#     data["dev"]["xray_paths"].tolist(), Ys[1], batch_size=32, shuffle=False
# )
# loaders["test"] = get_data_loader(
#     data["test"]["xray_paths"].tolist(), Ys[2], batch_size=32, shuffle=False
# )

# %% [markdown]
# As an example, a single datapoint yields an image like this:

# %%
# import matplotlib.pyplot as plt

# img, label = loaders["train"].dataset[0]

# plt.figure()
# plt.imshow(img[0, :, :], cmap="gray")
# plt.title("Example X-ray Image")
# ax = plt.axis("off")

# %% [markdown]
# Now that our `DataLoaders` are set up, it is a simple matter to define and train our CNN model.
#
# Note: While this will run if you do not have a CUDA-based GPU available (and will automatically detect it if you do), it will proceed *much* faster if you have one!  CPU-only per-epoch training time is ~ 15 minutes, while with a Titan X it is approximately 30 s!

# %%
# # Defining network parameters
# num_classes = 2
# pretrained = True
# train_args = [loaders["train"]]
# init_args = [[num_classes]]

# # Defining device variable
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Initializing input module
# input_module = cnn_model
# init_kwargs = {
#     "input_module": input_module,
#     "skip_head": True,
#     "input_relu": False,
#     "input_batchnorm": False,
#     "device": device,
#     "seed": 1701,
# }
# train_kwargs = {"n_epochs": 5, "progress_bar": True}

# # Setting up logger and searcher
# searcher = RandomSearchTuner(
#     EndModel,
#     **log_config,
#     log_writer_class=TensorBoardWriter,
#     validation_metric="accuracy",
#     seed=1701,
# )

# # Training weakly supervised model
# weakly_supervised_model = searcher.search(
#     search_space,
#     loaders["dev"],
#     train_args=train_args,
#     init_args=init_args,
#     init_kwargs=init_kwargs,
#     train_kwargs=train_kwargs,
#     max_search=tuner_config["max_search"],
#     clean_up=False,
# )

# %% [markdown]
# We can evaluate this model below, and see that we've learned some useful signal!  Remember that an Area Under the Receiver Operating Characteristic (ROC-AUC) score represents the probability across all possible cutoffs of ranking an abnormal example higher than a normal example.  If we've learned nothing useful, this value would be 0.5.
#
# You should expect a value just above 0.70 for this training run.

# %%
# # Evaluating model
# print(f"Evaluating Weakly Supervised Model")
# scores = weakly_supervised_model.score(loaders["test"], metric=["roc-auc"])

# %% [markdown]
# ## Step 7: Comparing to a Fully Supervised End Model

# %% [markdown]
# Because we have ground-truth labels for the entire dataset in this case (the OpenI dataset comes with these labels, which require physicians to label thousands of images!), we can compare how well our weakly supervised model does with the performance we achieve from a fully supervised model.  This is a similar analysis to that performed in our 2019 Nature submission.
#
# Executing this requires a simple change to the training dataloader to provide it with ground-truth labels.

# %%
# # Updating logging config
# log_config = {"log_dir": "./run_logs", "run_name": "openi_demo_fs"}


# # Creating dataloader with ground truth training labels
# loaders["full_train"] = get_data_loader(
#     data["train"]["xray_paths"].tolist(), Ys[0], batch_size=32, shuffle=True
# )
# train_args = [loaders["full_train"]]

# # Setting up logger and searcher
# searcher = RandomSearchTuner(
#     EndModel,
#     **log_config,
#     log_writer_class=TensorBoardWriter,
#     validation_metric="accuracy",
#     seed=1701,
# )

# # Training
# fully_supervised_model = searcher.search(
#     search_space,
#     loaders["dev"],
#     train_args=train_args,
#     init_args=init_args,
#     init_kwargs=init_kwargs,
#     train_kwargs=train_kwargs,
#     max_search=tuner_config["max_search"],
#     clean_up=False,
# )

# %% [markdown]
# Now, we can evaluate the weakly and fully supervised models, observing that they achieve similar Area Under the Receiver Operating Characteristic (ROC-AUC) scores.  Note that due to the small size of the dataset and that we are not tuning the cutoff for a particular performance score, we report ROC-AUC in this demo.

# %%
# # Evaluating weakly model
# print(f"Evaluating Weakly Supervised Model")
# weakly_supervised_scores = weakly_supervised_model.score(
#     loaders["test"], metric=["roc-auc"], print_confusion_matrix=False
# )

# # Evaluating fully supervised model
# print(f"Evaluating Fully Supervised Model")
# fully_supervised_scores = fully_supervised_model.score(
#     loaders["test"], metric=["roc-auc"], print_confusion_matrix=False
# )

# %% [markdown]
# If the models have trained successfully, you should observe that the weakly and fully supervised models both achieve ROC-AUC scores around 0.70.  This indicates that the weak labels we created using our labeling functions over the text have successfully allowed us to train a CNN model that performs very similarly to one trained using ground truth, but *without having to label thousands of images*.
#
# Congratulations! You've just trained a deep learning model using cross-modal data programming!  We hope this demo is helpful in your research, and check for updates to Snorkel and Snorkel MeTaL at [snorkel.stanford.edu](snorkel.stanford.edu)!
