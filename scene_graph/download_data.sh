#!/bin/bash
mkdir -p data
cd data

# download and unzip metadata and annotations.
wget https://www.dropbox.com/s/bnfhm6kt9xumik8/vrd.zip
unzip vrd.zip

# Delete the zip files.
rm vrd.zip
cd VRD

# Download and unzip images.
wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
unzip sg_dataset.zip
rm sg_dataset.zip

cd ../..

mkdir -p data/glove
cd data/glove

wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip

# Delete the zip files.
rm  glove.6B.zip