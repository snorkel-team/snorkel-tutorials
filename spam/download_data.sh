# Execute from root of snorkel-tutorials/
# Download data,

DATA_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"

cd spam
mkdir -p raw_data
wget $DATA_URL -O data.zip
mv data.zip raw_data/
cd raw_data
unzip data.zip
rm data.zip
rm -rf __MACOSX