#!/bin/bash

set -ex
set -o pipefail

SCRIPT_DIR=$(dirname "$0")
BUILD_DIR=($SCRIPT_DIR/../build)
TUTORIAL_DIR=$1

python $SCRIPT_DIR/check_notebooks.py $TUTORIAL_DIR
mkdir -p $BUILD_DIR
jupyter nbconvert $TUTORIAL_DIR/*.ipynb --to html --output-dir $BUILD_DIR
