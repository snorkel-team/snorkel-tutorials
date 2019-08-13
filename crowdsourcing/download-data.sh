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
    wget https://raw.githubusercontent.com/snorkel-team/snorkel/master/tutorials/crowdsourcing/data/weather-non-agg-DFE.csv -P data
    wget https://d1p17r2m4rzlbo.cloudfront.net/wp-content/uploads/2016/03/weather-evaluated-agg-DFE.csv -P data
fi
