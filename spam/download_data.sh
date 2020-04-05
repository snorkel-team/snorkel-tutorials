#!/bin/bash
set -euxo pipefail

# Check that we are running from the right directory.
if [ ! "${PWD##*/}" = "spam" ]; then
    echo "Script must be run from spam directory" >&2
    exit 1
fi

FILES=( "Youtube01-Psy.csv" "Youtube02-KatyPerry.csv" "Youtube03-LMFAO.csv" "Youtube04-Eminem.csv" "Youtube05-Shakira.csv" )
DATA_URL="http://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"
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
    curl $DATA_URL > data.zip
    mv data.zip data/
    cd data
    unzip data.zip
    rm data.zip
    rm -rf __MACOSX
    cd ..
fi
