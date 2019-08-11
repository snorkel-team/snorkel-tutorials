#!/bin/bash
set -euxo pipefail

# Check that we are running from the right directory.
if [ ! "${PWD##*/}" = "spouse" ]; then
    echo "Script must be run from spouse directory" >&2
    exit 1
fi

DATA_URL="https://www.dropbox.com/s/jmrvyaqew4zp9cy/spouse_data.zip"
FILES=( "train_data.pkl" "dev_data.pkl" "test_data.pkl" "dbpedia.pkl" )
RELOAD=false

# Check if at least any file is missing. If so, reload all data.
for filename in "${FILES[@]}"
do
    if [ ! -e "data/$filename" ]; then
        RELOAD=true
    fi
done

if [ "$RELOAD" = true ]; then
    if [ -d "data/" ]; then rm -Rf "data/"; fi
    mkdir -p data
    wget $DATA_URL -O data.zip
    cd data/
    unzip ../data.zip
    rm ../data.zip
    rm trained_spouse_model
fi
