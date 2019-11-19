#!/bin/bash
set -euxo pipefail

# Check that we are running from the right directory.
if [ ! "${PWD##*/}" = "visual_relation" ]; then
    echo "Script must be run from visual_relation directory" >&2
    exit 1
fi

DIRS=("glove" "VRD/sg_dataset")

RELOAD=false
# Check if at least any file is missing. If so, reload all data.
for directory_name in "${DIRS[@]}"
do
    if [ ! -d "data/$directory_name" ]; then
        RELOAD=true
    fi
done

ANNOTATIONS_URL="https://www.dropbox.com/s/bnfhm6kt9xumik8/vrd.zip"
IMAGES_URL="http://imagenet.stanford.edu/internal/jcjohns/visual_relations/sg_dataset.zip"
GLOVE_URL="http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"

if [ "$RELOAD" = true ]; then
    if [ -d "data" ]; then rm -Rf "data"; fi
    mkdir -p data
    cd data

    # download and unzip metadata and annotations
    echo "Downloading full VRD dataset..."
    wget $ANNOTATIONS_URL
    unzip vrd.zip

    # Delete the zip files.
    rm vrd.zip
    cd VRD

    # Download and unzip all images
    wget $IMAGES_URL
    unzip sg_dataset.zip
    rm sg_dataset.zip
    cd ..

    mkdir -p glove
    cd glove

    wget $GLOVE_URL
    unzip glove.6B.zip

    # Delete the zip files
    rm  glove.6B.zip
    cd ../..
fi

