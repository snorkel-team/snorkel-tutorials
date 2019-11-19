from pyspark.sql import Row
from snorkel.labeling.lf import labeling_function
from snorkel.labeling.lf.nlp_spark import spark_nlp_labeling_function
from snorkel.preprocess import preprocessor

from drybell_lfs import load_celebrity_knowledge_base

ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1


@preprocessor()
def combine_text(x):
    return Row(title=x.title, body=x.body, article=f"{x.title} {x.body}")


@spark_nlp_labeling_function(text_field="article", pre=[combine_text])
def article_mentions_person(x):
    for ent in x.doc.ents:
        if ent.label_ == "PERSON":
            return ABSTAIN
    return NEGATIVE


@spark_nlp_labeling_function(
    text_field="article",
    pre=[combine_text],
    resources=dict(celebrity_knowledge_base=load_celebrity_knowledge_base()),
)
def person_in_db(x, celebrity_knowledge_base):
    for ent in x.doc.ents:
        if ent.label_ == "PERSON" and ent.text.lower() in celebrity_knowledge_base:
            return POSITIVE
    return ABSTAIN


@labeling_function()
def body_contains_fortune(x):
    return POSITIVE if "fortune" in x.body else ABSTAIN
