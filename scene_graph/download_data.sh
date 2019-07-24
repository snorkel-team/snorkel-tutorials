# Execute from snorkel-tutorials/scene_graph
# Download data

ANNOTATIONS_URL="https://www.dropbox.com/s/bnfhm6kt9xumik8/vrd.zip"
IMAGES_URL="http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip"
SAMPLE_IMAGES_URL="https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection/trunk/samples"
GLOVE_URL="http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"

if [ ! -d "data" ]; then
    mkdir -p data
    cd data

    # download and unzip metadata and annotations
    wget $ANNOTATIONS_URL
    unzip vrd.zip

    # Delete the zip files.
    rm vrd.zip
    cd VRD

    if [ "$TRAVIS" = "true" ]; then
        # Download and unzip sample images
        mkdir sg_dataset
        cd sg_dataset
        svn checkout $SAMPLE_IMAGES_URL
        cd ../..
    else
        # Download and unzip all images
        wget $IMAGES_URL
        unzip sg_dataset.zip
        rm sg_dataset.zip
        cd ../..
    fi

    mkdir -p data/glove
    cd data/glove

    wget $GLOVE_URL
    unzip glove.6B.zip

    # Delete the zip files
    rm  glove.6B.zip
fi

