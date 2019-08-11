# Execute from snorkel-tutorials/
# Download data,

ANNOTATIONS_URL="https://www.dropbox.com/s/bnfhm6kt9xumik8/vrd.zip"
IMAGES_URL="http://imagenet.stanford.edu/internal/jcjohns/visual_relations/sg_dataset.zip"
SAMPLE_IMAGES_URL="https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection.git"
GLOVE_URL="http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"

if [ ! -d "visual_relation/data" ]; then
    mkdir -p visual_relation/data
    cd visual_relation/data

    # download and unzip metadata and annotations
    wget $ANNOTATIONS_URL
    unzip vrd.zip

    # Delete the zip files.
    rm vrd.zip
    cd VRD

    # Download and unzip all images
    wget $IMAGES_URL
    unzip sg_dataset.zip
    rm sg_dataset.zip
    cd ../../..

    mkdir -p visual_relation/data/glove
    cd visual_relation/data/glove

    wget $GLOVE_URL
    unzip glove.6B.zip

    # Delete the zip files
    rm  glove.6B.zip
    cd ../../..
fi

