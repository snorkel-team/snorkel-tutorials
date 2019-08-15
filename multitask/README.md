# Multitask Learning Tutorial
Multi-task learning, or training a single model on multiple tasks, is becoming a standard tool for the modern ML practioner (see Ruder's [survey](http://ruder.io/multi-task/) from 2017 for a nice overview).
It often leads to computational gains (one model performing many tasks takes up less memory and storage) as well as performance gains (learning to do well on a related _auxiliary_ task can improve the model's ability on the _primary_ task).
While the primary purpose of the Snorkel project is to support training data creation and management, it also comes with a PyTorch-based modeling framework intended to support flexible multi-task learning (e.g. slice-aware models).
In this tutorial, we introduce the basic interfaces and flow of multi-task learning tools within Snorkel.

