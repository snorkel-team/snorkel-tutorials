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
    if [ ! -d "scene_graph/data/$directory_name" ]; then
        RELOAD=true
    fi
done

ANNOTATIONS_URL="https://www.dropbox.com/s/bnfhm6kt9xumik8/vrd.zip"
IMAGES_URL="http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip"
SAMPLE_IMAGES_URL="https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection.git"
GLOVE_URL="http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"

if [ ! -d "scene_graph/data" ]; then
    if [ -d "scene_graph/data/" ]; then rm -Rf "scene_graph/data/"; fi
    mkdir -p scene_graph/data
    cd scene_graph/data

    # download and unzip metadata and annotations
    wget $ANNOTATIONS_URL
    unzip vrd.zip

    # Delete the zip files.
    rm vrd.zip
    cd VRD

    # if [ "$TRAVIS" = "true" ]; then
    #     # Download and unzip sample images
    #     mkdir sg_dataset
    #     cd sg_dataset
    #     git clone $SAMPLE_IMAGES_URL
    #     mv Visual-Relationship-Detection/samples ./
    #     rm -r Visual-Relationship-Detection
    #     cd ../..
    # else
    #     # Download and unzip all images
    #     wget $IMAGES_URL
    #     unzip sg_dataset.zip
    #     rm sg_dataset.zip
    #     cd ../../..
    # fi

    # Download and unzip all images
    mkdir sg_dataset
    cd sg_dataset
    git clone $SAMPLE_IMAGES_URL
    mv Visual-Relationship-Detection/samples ./
    rm -r Visual-Relationship-Detection
    cd ../../../..

    mkdir -p scene_graph/data/glove
    cd scene_graph/data/glove

    wget $GLOVE_URL
    unzip glove.6B.zip

    # Delete the zip files
    rm  glove.6B.zip
    cd ../../..
fi

