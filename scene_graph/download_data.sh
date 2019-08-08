# Execute from snorkel-tutorials/
set -euxo pipefail
trap 'rm -r scene_graph/data/' ERR  # Clean up in case of error.

ANNOTATIONS_URL="https://www.dropbox.com/s/bnfhm6kt9xumik8/vrd.zip"
IMAGES_URL="http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip"
SAMPLE_IMAGES_URL="https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection.git"
GLOVE_URL="http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"

if [ ! -d "scene_graph/data" ]; then
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

