# Execute from snorkel-tutorials/spam/
set -euxo pipefail
trap 'rm -r data/ data.zip' ERR  # Clean up in case of error.

DATA_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"

if [ ! -d "data" ]; then
    mkdir -p data
    wget $DATA_URL -O data.zip
    mv data.zip data/
    cd data
    unzip data.zip
    rm data.zip
    rm -rf __MACOSX
    cd ..
fi
