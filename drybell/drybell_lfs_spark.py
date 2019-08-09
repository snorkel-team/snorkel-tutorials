from pyspark.sql import Row
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.preprocess.spark import make_spark_preprocessor

ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1


@preprocessor()
def combine_text_spark(x):
    return Row(article=f"{x.title} {x.body}", title=x.title, body=x.body)


spacy_preprocessor = SpacyPreprocessor(
    text_field="article", doc_field="doc", pre=[combine_text_spark], memoize=True
)
spacy_preprocessor_spark = make_spark_preprocessor(spacy_preprocessor)


@labeling_function(pre=[spacy_preprocessor_spark])
def article_mentions_person(x):
    for ent in x.doc.ents:
        if ent.label_ == "PERSON":
            return ABSTAIN
    return NEGATIVE


def load_celebrity_knowledge_base(path="drybell/data/celebrity_knowledge_base.txt"):
    with open(path, "r") as f:
        return f.read().splitlines()


@labeling_function(
    pre=[spacy_preprocessor_spark],
    resources=dict(celebrity_knowledge_base=load_celebrity_knowledge_base()),
)
def person_in_db(x, celebrity_knowledge_base):
    for ent in x.doc.ents:
        if ent.label_ == "PERSON" and ent.text.lower() in celebrity_knowledge_base:
            return POSITIVE
    return ABSTAIN
