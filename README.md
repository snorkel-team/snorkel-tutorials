# Snorkel Tutorials
A collection of tutorials for [Snorkel](http://snorkel.org).

## Tutorials
The Snorkel tutorials are grouped by application, with some applications having multiple associated notebooks in their directory.
* `spam`: Is this YouTube comment spam?
* `spouse`: Does this sentence imply that the two marked people are spouses?
* `visual` (Visual Relationship Detection): Is object A riding object B, carrying it, or neither?
* `weather`: Is this tweet about the weather expressing a positive, negative or neutral sentiment?
* `mtl` (Multi-Task Learning): A synthetic task demonstrating the native Snorkel multi-task classifier API

See the [Tutorials Index](#tutorials-index) for a listing of which tutorials demonstrate which task types, techniques, and integrations.

## Getting Started
Start with the `spam` tutorial for a gentle introduction to the concepts and classes of Snorkel.
All other tutorials assume that you have already completed that tutorial and are familiar with its concepts.

To run a tutorial in a jupyter notebook, run the following commands, which create a virtual environment, install requirements, create an ipython kernel, and launch Jupyter:
```bash
TUTORIAL="spam"  # Change this to the name of the directory for the tutorial you want

# Create virtual env
VIRTUALENV=".env_${TUTORIAL}"
virtualenv $VIRTUALENV
source $VIRTUALENV/bin/activate  # Activate the created virtual environment
pip3 install -r requirements.txt  # Requirements shared among all tutorials
cd $TUTORIAL
pip3 install -r requirements.txt  # Requirements specific to this tutorial

# Launch Jupyter
jupyter notebook

# To deactivate the virtual environment when you are done, run `deactivate`.
```
Then, in the browser tab that opens, select the notebook you would like to run.

Alternatively, to run the tutorial as a script, skip the Jupyter launch command and run the tutorial's `.py` file directly (e.g. `python spam_tutorial.py`).


## <a name="tutorials-index"> Tutorials Index </a>
Here we provide an index pointing to different available tutorials by their task type, techniques, and integrations.
* Task
    * Text Classification (Text): `spam`, `weather`
    * Relation Extraction (Text): `spouse`
    * Visual Relationship Detection (Image): `visual`
* Techniques
    * Labeling with Labeling Functions (LFs): `spam`, `spouse`, `visual`, `weather`
    * Augmentation with Transformation Functions (TFs): `spam`
    * Monitoring with Slicing Functions (SFs): `spam`
    * Using Crowdworker Labels: `weather`
    * Multi-Task Learning (MTL): `mtl`, `visual`, `spam`
* Integrations
    * TensorFlow/Keras: `spam`
    * Scikit-Learn: `spam`
    * PyTorch: `spam`
