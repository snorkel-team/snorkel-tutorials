#!/bin/bash
set -euxo pipefail

# Check that we are running from the right directory.
if [ ! "${PWD##*/}" = "crowdsourcing" ]; then
    echo "Script must be run from crowdsourcing directory" >&2
    exit 1
fi

FILES=( "weather-non-agg-DFE.csv" "weather-evaluated-agg-DFE.csv" )
RELOAD=false

# Check if at least any file is missing. If so, reload all data.
for filename in "${FILES[@]}"
do
    if [ ! -e "data/$filename" ]; then
        RELOAD=true
    fi
done

if [ "$RELOAD" = "true" ]; then
    if [ -d "data/" ]; then rm -Rf "data/"; fi
    mkdir -p data
    wget https://www.dropbox.com/s/94d2wsrrwh1ioyd/weather-non-agg-DFE.csv -P data
    wget https://www.dropbox.com/s/upz3ijyp7rztse6/weather-evaluated-agg-DFE.csv -P data
fi
