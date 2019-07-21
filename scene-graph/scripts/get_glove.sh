#!/bin/bash
mkdir -p data/glove
cd data/glove

wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip

# Delete the zip files.
rm  glove.6B.zip
