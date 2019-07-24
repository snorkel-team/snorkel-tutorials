# Snorkel Tutorials
A collection of tutorials for [Snorkel](http://snorkel.org).

## Tutorials
The Snorkel tutorials are grouped by application, with some applications having multiple associated notebooks in their directory.
* `Spam`: Is this YouTube comment spam?
* `Spouse`: Are these two people mentioned in a sentence married?
* `VRD` (Visual Relationship Detection): Is object A riding object B, carrying it, or neither?
* `Weather`: Is this tweet about the weather expressing a positive, negative or neutral sentiment?
* `MTL` (Multi-Task Learning): A synthetic task demonstrating the native Snorkel multi-task classifier API

See the [Tutorials Index](#tutorials-index) for a listing of which tutorials demonstrate which task types, techniques, and integrations.

## Getting Started
Start with the `Spam` tutorial for a gentle introduction to the concepts and classes of Snorkel.
All other tutorials assume that you have already completed that tutorial and are familiar with its concepts.

To run a tutorial in a jupyter notebook, run the following commands, which create a virtual environment, install requirements, create an ipython kernel, and launch Jupyter:
```
TUTORIAL="spam"  # Change this to the name of the directory for the tutorial you want
VIRTUALENV=".env_${TUTORIAL}"
virtualenv $VIRTUALENV
source $VIRTUALENV/bin/activate
pip install -r requirements.txt  # Requirements shared among all tutorials
cd $TUTORIAL
pip install -r requirements.txt  # Requirements specific to this tutorial
ipython kernel install --user --name=$VIRTUALENV
jupyter notebook
```
Then, in the browser tab that opens, select the notebook you would like to run and confirm that the appropriately named ipython kernel (e.g., `.env_spam`) is displayed in the upper right corner. If not, go to `Kernel` > `Change Kernel` > and select the appropriate environment.

Alternatively, to run the tutorial as a script, skip the last two steps of the setup instructions above (`ipython kernel` and `jupyter notebook`) and run the tutorial's `.py` file directly (e.g. `python spam_tutorial.py`).


## <a name="tutorials-index"> Tutorials Index </a>
Here we provide an index pointing to different available tutorials by their task type, domain type, integrations, ... TBD.
* Task
    * Text Classification (Text): `Spam`, `Weather`
    * Relation Extraction (Text): `Spouse`
    * Visual Relationship Detection (Image): `VRD`
* Techniques
    * Labeling with Labeling Functions (LFs): `Spam`, `Spouse`, `VRD`, `Weather`
    * Augmentation with Transformation Functions (TFs): `Spam`
    * Monitoring with Slicing Functions (SFs): `Spam`
    * Using Crowdworker Labels: `Weather`
    * Multi-Task Learning (MTL): `MTL`
* Classifiers
    * TensorFlow/Keras: `Spam`
    * Scikit-Learn: `Spam`
    * PyTorch: `Spam`
