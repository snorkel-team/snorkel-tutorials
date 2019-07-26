#!/bin/bash
TUTORIAL=$1

if [ -z $TUTORIAL ]; then
    echo "You must provide the name of the tutorial whose environment you are creating (e.g., spam)."
    exit 1
fi

# Create virtual env if it does not already exist
VIRTUALENV=".env_${TUTORIAL}"
if [ ! -d $VIRTUALENV ]; then
    echo "Creating virtual environment for tutorial $TUTORIAL."
    pip install virtualenv
    virtualenv $VIRTUALENV
fi

# Activate virtual environment
echo "Activating the virtual envrionment for tutorial $TUTORIAL."
source $VIRTUALENV/bin/activate  # Activate the created virtual environment
