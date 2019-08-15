# Crowdsourcing Tutorial
In this tutorial, we provide a simple walkthrough of how to use Snorkel in conjunction with crowdsourcing to create a training set for a sentiment analysis task.
We use a dataset where we have crowdsourced labels for about half of the training data points.
The crowdsourced labels are fairly accurate, but do not cover the entire training dataset, nor are they available for the test set or during inference.
To make up for their lack of training set coverage, we combine crowdsourced labels with heuristic labeling functions to increase the number of training labels we have.
Then we combine the training labels using a `LabelModel` and use its outputs to train a deep learning model which can be applied to new, unseen data to automatically make predictions.
