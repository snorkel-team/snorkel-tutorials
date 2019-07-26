#!/bin/bash
TUTORIAL=$1

if [ -z $TUTORIAL ]; then
    echo "You must provide the name of the tutorial you want to launch (e.g., spam)"
    exit 1
fi

bash create_environment.sh $TUTORIAL
bash install_requirements.sh $TUTORIAL
jupyter notebook