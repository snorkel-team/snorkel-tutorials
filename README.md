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

Step one is cloning this repo.
If you're looking to quickly get started with a tutorial, we recommend using our [Docker setup](#docker).
If you want to install things yourself, you can follow our [installation steps](#install) below instead.

### <a name="docker"> Running with Docker </a>

We've included a Docker setup for our tutorials to make setup easy.
First, make sure you have [Docker installed](https://docs.docker.com/install/) on your machine.
To build and run a Docker image for a tutorial, use `scripts/docker_launch.py` with the `--build` flag.
For example, run the following for the `spam` tutorial:

```bash
python3 scripts/docker_launch.py spam --build
```

This will make a Jupyter notebook server available on port 8888
(you can change the port with the `--port` command line option)
and print out a link you can follow to access the browser interface.
In your browser, open a `.ipynb` file you would like to run &mdash;
such as `spam_tutorial.ipynb` &mdash; and execute the cells in sequence.

Once an image has been built, you can run it without the `--build` flag:

```bash
python3 scripts/docker_launch.py spam
```

### <a name="install"> Installing yourself </a>

Running a tutorial has three required steps if you're installing yourself:

1. Installing repo-wide requirements
1. Installing tutorial-specific requirements
1. Launching a Jupyter notebook server or executing as a script

We recommend installing requirements in a virtual environment using [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or [`conda`](https://docs.conda.io/en/latest/).
For example, if you use pip, first activate your virtualenv if you're using one, then run:

```bash
# Install requirements (both shared and tutorial-specific)
pip3 install -r requirements.txt
pip3 install -r spam/requirements.txt

# Launch the Jupyter notebook interface
jupyter notebook
```
Then in the browser tab that opens, navigate to a `.ipynb` file you would like to run &mdash;
such as `spam/spam_tutorial.ipynb` &mdash; and execute the cells in sequence.

Alternatively, you can run the tutorial as a script by calling `python3` on the corresponding `.py` file directly (e.g. `python3 spam_tutorial.py`).


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
