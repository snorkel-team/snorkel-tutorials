FROM python:3.6-slim-stretch
RUN apt-get update \
    && apt-get install -y \
    apt-utils \
    build-essential \
    git \
    wget \
    curl \
    unzip

WORKDIR /app
COPY . /app
ARG TUTORIAL
RUN wget -O snorkel-requirements.txt \
    https://raw.githubusercontent.com/snorkel-team/snorkel/master/requirements.txt \
    && pip3 install -r $TUTORIAL/requirements.txt \
    && pip3 install -r requirements.txt \
    && pip3 install -r snorkel-requirements.txt \
    && python3 -m spacy download en_core_web_sm

WORKDIR $TUTORIAL
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
