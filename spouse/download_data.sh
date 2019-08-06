# Execute from root of snorkel-tutorials/

DATA_URL="https://www.dropbox.com/s/jmrvyaqew4zp9cy/spouse_data.zip"

if [ ! -d "data" ]; then
    mkdir -p data
    wget $DATA_URL -O data.zip
    cd data/
    unzip ../data.zip
    rm ../data.zip
fi
