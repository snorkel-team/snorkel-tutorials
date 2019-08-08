#!/bin/bash
set -euxo pipefail

if [ ! -d "data" ]; then
    mkdir -p data
    wget https://raw.githubusercontent.com/HazyResearch/snorkel/master/tutorials/crowdsourcing/data/weather-non-agg-DFE.csv -P data
    wget https://d1p17r2m4rzlbo.cloudfront.net/wp-content/uploads/2016/03/weather-evaluated-agg-DFE.csv -P data
fi
