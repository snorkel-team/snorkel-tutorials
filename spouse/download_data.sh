# Execute from root of snorkel-tutorials/
set -euxo pipefail
trap 'rm -r data/ data.zip' ERR  # Clean up in case of error.

DATA_URL="https://www.dropbox.com/s/jmrvyaqew4zp9cy/spouse_data.zip"

if [ ! -d "data" ]; then
    mkdir -p data
    wget $DATA_URL -O data.zip
    cd data/
    unzip ../data.zip
    rm ../data.zip
    rm trained_spouse_model
fi
