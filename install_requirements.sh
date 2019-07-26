#!/bin/bash
TUTORIAL=$1

if [ -z $TUTORIAL ]; then
    echo "You must provide the name of the tutorial whose requirements you are installing (e.g., spam)."
    exit 1
fi

# Install requirements
echo "Installing snorkel and other shared tutorial requirements from snorkel-tutorials/requirements.txt (this may take a few minutes)."
pip install -r requirements.txt  # Install requirements shared among all tutorials
cd $TUTORIAL
echo "Installing requirements specific to this tutorial from snorkel-tutorials/$TUTORIAL/requirements.txt (this may take a few minutes)."
pip install -r requirements.txt  # Install requirements specific to this tutorial
echo "Finished installing requirements"