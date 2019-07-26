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

We recommend that all users **start with the `spam` tutorial** for a gentle introduction to the concepts and classes of Snorkel.  
All other tutorials assume that you have already completed that tutorial and are familiar with its concepts.

## Getting Started
If you just want to get up and running, run the following command to:

0. Create a virtual environment and activate it
1. Install requirements
2. Launch the Jupyter notebook interface
```bash
bash launch.sh spam  # Change `spam` to the name of the tutorial to run
```
Then, in the browser tab that opens, navigate to a `*.ipynb` you would like to run, such as `snorkel-tutorials/spam/spam_tutorial.ipynb`, and click to open it.

Alternatively, you can go through these steps individually using the commands below.

### 0. Create Virtual Environment (Optional)
To avoid conflicts with other packages and projects you may have on your machine, we recommend creating a separate virtual environment for each tutorials. 
This can be done with the following command, which installs `virtualenv`, creates a new virtual environment, and activates it. 
To deactivate when you're done, simply type `deactivate`.
```bash
# Change 'spam' to the name of the tutorial whose environment you are creating.
bash create_environment.sh spam
```

### 1. Install Requirements
Each tutorial depends on some requirements common to all tutorials (`requirements.txt`), and some specific to itself (e.g., `spam/requirements.txt`). 
To install both, run the following command.
```bash
# Change 'spam' to the name of the tutorial whose requirements you are installing.
bash install_requirements.sh spam
```

### 2. Launch Jupyter
To launch [Jupyter](https://jupyter.org/), run the following command.
```bash
jupyter lab
```
Then, in the browser tab that opens, navigate to a `*.ipynb` you would like to run, such as `snorkel-tutorials/spam/spam_tutorial.ipynb`, and click to open it.

Alternatively, you can run the tutorial as a script by calling python on the corresponding `.py` file directly (e.g. `python spam_tutorial.py`).


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
