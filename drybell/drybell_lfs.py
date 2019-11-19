from snorkel.labeling.lf import labeling_function
from snorkel.labeling.lf.nlp import nlp_labeling_function
from snorkel.preprocess import preprocessor

ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1


@preprocessor()
def combine_text(x):
    x.article = f"{x.title} {x.body}"
    return x


@nlp_labeling_function(text_field="article", pre=[combine_text])
def article_mentions_person(x):
    for ent in x.doc.ents:
        if ent.label_ == "PERSON":
            return ABSTAIN
    return NEGATIVE


def load_celebrity_knowledge_base(path="drybell/data/celebrity_knowledge_base.txt"):
    with open(path, "r") as f:
        return f.read().splitlines()


@nlp_labeling_function(
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
