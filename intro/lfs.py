from snorkel.labeling.lf import labeling_function
import re
from snorkel.labeling.preprocess import preprocessor
from textblob import TextBlob


HAM = 0
SPAM = 1
ABSTAIN = -1


### KEYWORD LFS
@labeling_function()
def lf_keyword_my(x):
    """Many spam comments talk about 'my channel', 'my video', etc."""
    return SPAM if "my" in x.text.lower() else ABSTAIN


@labeling_function()
def lf_subscribe(x):
    """Spam comments ask users to subscribe to their channels."""
    return SPAM if "subscribe" in x.text else ABSTAIN


@labeling_function()
def lf_link(x):
    """Spam comments post links to other channels."""
    return SPAM if "http" in x.text.lower() else ABSTAIN


@labeling_function()
def lf_please(x):
    """Spam comments make requests rather than commenting."""
    return (
        SPAM if any([word in x.text.lower() for word in ["please", "plz"]]) else ABSTAIN
    )


@labeling_function()
def lf_song(x):
    """Ham comments actually talk about the video's content."""
    return HAM if "song" in x.text.lower() else ABSTAIN


### REGEX LFS
@labeling_function()
def lf_regex_check_out(x):
    """Spam comments say 'check out my video', 'check it out', etc."""
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN


### HEURISTIC LFS
@labeling_function()
def lf_short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return HAM if len(x.text.split()) < 5 else ABSTAIN


### THIRD-PARTY MODEL LFS
@preprocessor(memoize=True)
def text_blob_sentiment(x):
    x.sentiment = TextBlob(x.text).sentiment
    return x


@labeling_function(preprocessors=[text_blob_sentiment])
def lf_textblob_polarity(x):
    return HAM if x.sentiment.polarity > 0.3 else ABSTAIN


@labeling_function(preprocessors=[text_blob_sentiment])
def lf_textblob_subjectivity(x):
    return HAM if x.sentiment.subjectivity > 0.9 else ABSTAIN


### ALL LFS
lfs = [
    lf_keyword_my,
    lf_subscribe,
    lf_link,
    lf_please,
    lf_song,
    lf_regex_check_out,
    lf_short_comment,
    lf_textblob_polarity,
    lf_textblob_subjectivity,
]