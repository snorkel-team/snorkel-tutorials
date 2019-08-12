#!/bin/bash
set -euxo pipefail

# Check that we are running from the right directory.
if [ ! "${PWD##*/}" = "snorkel-tutorials" ]; then
    echo "Script must be run from snorkel-tutorials directory" >&2
    exit 1
fi

DIRS=("glove" "VRD/sg_dataset/samples")

# Check if at least any file is missing. If so, reload all data.
for directory_name in "${DIRS[@]}"
do
    if [ ! -d "visual_relation/data/$directory_name" ]; then
        RELOAD=true
    fi
done

ANNOTATIONS_URL="https://www.dropbox.com/s/bnfhm6kt9xumik8/vrd.zip"
IMAGES_URL="http://imagenet.stanford.edu/internal/jcjohns/visual_relations/sg_dataset.zip"
SAMPLE_IMAGES_URL="https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection.git"
GLOVE_URL="http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"

if [ ! -d "visual_relation/data" ]; then
    if [ -d "visual_relation/data/" ]; then rm -Rf "visual_relation/data/"; fi
    mkdir -p visual_relation/data
    cd visual_relation/data

    # download and unzip metadata and annotations
    wget $ANNOTATIONS_URL
    unzip vrd.zip

    # Delete the zip files.
    rm vrd.zip
    cd VRD

    # Download and unzip sample images
    mkdir sg_dataset
    cd sg_dataset
    git clone $SAMPLE_IMAGES_URL
    mv Visual-Relationship-Detection/samples ./
    rm -r Visual-Relationship-Detection
    cd ../../../..

    mkdir -p visual_relation/data/glove
    cd visual_relation/data/glove

    wget $GLOVE_URL
    unzip glove.6B.zip

    # Delete the zip files
    rm  glove.6B.zip
    cd ../../..
fi

