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
To run a tutorial, clone this repository and run the following commands (replacing `spam` with the name of the tutorial you want to run, if applicable). We recommend doing this in a virtual environment with e.g., `virtualenv` or `conda`.

```
# Install requirements (both shared and tutorial-specific)
pip3 install -r requirements.txt
pip3 install -r spam/requirements.txt

# Launch the Jupyter notebook interface
jupyter notebook
```
Then, in the browser tab that opens, navigate to a `.ipynb` file you would like to run, such as `snorkel-tutorials/spam/spam_tutorial.ipynb`, and click to open it. 
Then execute the cells in sequence.

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
