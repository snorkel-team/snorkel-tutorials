# Snorkel Tutorials
[![build](https://travis-ci.com/snorkel-team/snorkel-tutorials.svg?branch=master)](https://travis-ci.com/snorkel-team/snorkel-tutorials?branch=master)

A collection of tutorials for [Snorkel](https://github.com/snorkel-team/snorkel).
For more information, visit the [Snorkel website](http://snorkel.org).

## Tutorials
The Snorkel tutorials are grouped by application, with some applications having multiple associated notebooks in their directory.
* `spam`: Is this YouTube comment spam?
* `spouse`: Does this sentence imply that the two marked people are spouses?
* `visual_relation`: Is object A riding object B in the image, carrying it, or neither?
* `crowdsourcing`: Is this tweet about the weather expressing a positive, negative or neutral sentiment?
* `multitask` (Multi-Task Learning): A synthetic task demonstrating the native Snorkel multi-task classifier API
* `recsys` (Recommender Systems): Will this user read and like this book?
* [`drybell`](https://ai.googleblog.com/2019/03/harnessing-organizational-knowledge-for.html): Is a celebrity mentioned in this news article?

See the [Tutorials Index](#tutorials-index) for a listing of which tutorials demonstrate which task types, techniques, and integrations.
If you are interested in requesting or contributing a new tutorial, let us know on the Snorkel [community forum](https://spectrum.chat/snorkel/tutorials?tab=posts).

We recommend that all users **start with the `spam` tutorial** for a gentle introduction to the concepts and classes of Snorkel.
All other tutorials assume that you have already completed that tutorial and are familiar with its concepts.

## Getting Started

Step one is cloning this repo.

```bash
git clone https://github.com/snorkel-team/snorkel-tutorials.git
cd snorkel-tutorials
```

As with Snorkel, our tutorials require Python 3.6+.
If you're looking to quickly get started with a tutorial, we recommend using
our [Docker setup](#docker).
If you want to install things yourself using `pip` or `conda`, you can follow
our [installation steps](#install) below instead.

***A quick note for Windows users***

If you're using Windows, we highly recommend using the [Docker setup](#docker)
or the [Linux subsystem](https://docs.microsoft.com/en-us/windows/wsl/faq).
It can be tricky to get the installation right using application-specific shells
(e.g. the `conda` shell).
Additionally, the shell scripts included in this repo (such as those for
downloading datasets) use *nix-style commands.


### <a name="docker"> Running with Docker </a>

We've included a Docker setup for our tutorials to make setup easy.
First, make sure you have [Docker installed](https://docs.docker.com/install/) on your machine.
To build and run a Docker image for a tutorial, use `scripts/docker_launch.py` with the `--build` flag.
For example, run the following for the `spam` tutorial:

```bash
python3 scripts/docker_launch.py spam --build
```

Building a Docker image from scratch can take anywhere between 5 and
30 minutes depending on the machine you're using.
We're working on making prebuilt images available via DockerHub.

Once the image has been built, a Jupyter notebook server will be available
on port 8888 (you can change the port with the `--port` command line option)
and print out a link you can follow to access the browser interface.
In your browser, open a `.ipynb` file you would like to run &mdash;
such as `01_spam_tutorial.ipynb` &mdash; and execute the cells in sequence.

Once you've built a tutorial-specific image for the first time,
you can run it without the `--build` flag:

```bash
python3 scripts/docker_launch.py spam
```

### <a name="install"> Installing yourself </a>

Running a tutorial has three required steps if you're installing yourself:

1. Installing repo-wide requirements
1. Installing tutorial-specific requirements
1. Launching a Jupyter notebook server or executing as a script

We recommend installing requirements in a virtual environment using [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or [`conda`](https://docs.conda.io/en/latest/).

The following example commands show you how to install the requirements for the
`spam` tutorial, then launch a notebook server to run the tutorial.
To run a different tutorial, simply replace `spam` with the desired directory.

<details><summary><b>Installing with <tt>pip</tt></b></summary>
<p>

These commands assume that your Python version is 3.6+ and that the Python 3
version of `pip` is available as `pip3`.
It may be available as `pip` depending on how your system is configured.

```bash
# [OPTIONAL] Activate a virtual environment
pip3 install --upgrade virtualenv
virtualenv -p python3 .envspam
source .envspam/bin/activate

# Install requirements (both shared and tutorial-specific)
pip3 install -r requirements.txt
pip3 install -r spam/requirements.txt

# Launch the Jupyter notebook interface
jupyter notebook spam
```

</p>
</details>


<details><summary><b>Installing with <tt>conda</tt></b></summary>
<p>

These commands assume that your conda installation is Python 3.6+.

```bash
# [OPTIONAL] Activate a virtual environment
conda create --yes -n spam python=3.6
source activate spam

# Install requirements (both shared and tutorial-specific)
pip install environment_kernels
pip install -r requirements.txt
pip install -r spam/requirements.txt

# Launch the Jupyter notebook interface
jupyter notebook spam
```

Make sure to select the right kernel (`conda_spam`) when running the jupyter notebook.

</p>
</details>

Then in the browser tab that opens, navigate to a `.ipynb` file you would like
to run &mdash; such as `01_spam_tutorial.ipynb` &mdash; and execute the
cells in sequence.

Alternatively, you can run the tutorial as a script by calling `python3` on the corresponding `.py` file directly (e.g. `python3 spam/01_spam_tutorial.py`).
The `.py` source files are written in [Jupytext `percent` format](https://jupytext.readthedocs.io/en/latest/), and contain the same content as the notebooks.


## <a name="tutorials-index"> Tutorials Index </a>
Here we provide an index pointing to different available tutorials by their task type, techniques, and integrations.
* Task
    * Text Classification (Text): `spam`, `crowdsourcing`, `drybell`
    * Relation Extraction (Text): `spouse`
    * Visual Relationship Detection (Image): `visual_relation`
    * Recommender systems: `recsys`
* Techniques
    * Labeling with Labeling Functions (LFs): `spam`, `spouse`, `visual_relation`, `crowdsourcing`
    * Augmentation with Transformation Functions (TFs): `spam`
    * Monitoring with Slicing Functions (SFs): `spam`
    * Using Crowd Worker Labels: `crowdsourcing`
    * Multi-Task Learning (MTL): `multitask`, `visual_relation`, `spam`
* Integrations
    * TensorFlow/Keras: `spam`, `spouse`
    * Scikit-Learn: `spam`, `crowdsourcing`
    * PyTorch: `multitask`
    * Dask: `drybell`
    * Spark: `drybell`
