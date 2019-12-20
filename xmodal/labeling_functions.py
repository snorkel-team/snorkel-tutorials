import re
from snorkel.labeling import labeling_function

# Defining labels
ABSTAIN = -1
ABNORMAL = 1
NORMAL = 0

# Defining some global lists of useful terms
negative_inflection_words = ["but", "however", "otherwise"]

abnormal_mesh_terms = [
    "opacity",
    "cardiomegaly",
    "calcinosis",
    "hypoinflation",
    "calcified granuloma",
    "thoracic vertebrae",
    "degenerative",
    "hyperdistention",
    "catheters",
    "granulomatous",
    "nodule",
    "fracture" "surgical",
    "instruments",
    "emphysema",
]

words_indicating_normalcy = ["clear", "no", "normal", "unremarkable", "free", "midline"]

categories = [
    "normal",
    "opacity",
    "cardiomegaly",
    "calcinosis",
    "lung/hypoinflation",
    "calcified granuloma",
    "thoracic vertebrae/degenerative",
    "lung/hyperdistention",
    "spine/degenerative",
    "catheters, indwelling",
    "granulomatous disease",
    "nodule",
    "surgical instruments",
    "scoliosis",
    "osteophyte",
    "spondylosis",
    "fractures, bone",
]

# Defining useful regexes
reg_equivocation = re.compile(
    "unlikely|likely|suggests|questionable|concerning|possibly|potentially|could represent|may represent|may relate|cannot exclude|can't exclude|may be",
    re.IGNORECASE,
)


@labeling_function()
def LF_report_is_short(x):
    """
    Checks if report is short.
    """
    return NORMAL if len(x.text) < 280 else ABSTAIN


@labeling_function()
def LF_negative_inflection_words_in_report(x):
    return (
        ABNORMAL
        if any(word in x.text.lower() for word in negative_inflection_words)
        else ABSTAIN
    )


@labeling_function()
def LF_is_seen_or_noted_in_report(x):
    return (
        ABNORMAL
        if any(word in x.text.lower() for word in ["is seen", "noted"])
        else ABSTAIN
    )


@labeling_function()
def LF_disease_in_report(x):
    return ABNORMAL if "disease" in x.text.lower() else ABSTAIN


@labeling_function()
def LF_recommend_in_report(x):
    return ABNORMAL if "recommend" in x.text.lower() else ABSTAIN


@labeling_function()
def LF_mm_in_report(x):
    return ABNORMAL if any(word in x.text.lower() for word in ["mm", "cm"]) else ABSTAIN


@labeling_function()
def LF_abnormal_mesh_terms_in_report(x):
    if any(mesh in x.text.lower() for mesh in abnormal_mesh_terms):
        return ABNORMAL
    else:
        return ABSTAIN


@labeling_function()
def LF_consistency_in_report(x):
    """
    The words 'clear', 'no', 'normal', 'free', 'midline' in
    findings section of the report
    """
    report = x.text
    findings = report[report.find("FINDINGS:") :]
    findings = findings[: findings.find("IMPRESSION:")]
    sents = findings.split(".")

    num_sents_without_normal = ABSTAIN
    for sent in sents:
        sent = sent.lower()
        if not any(word in sent for word in words_indicating_normalcy):
            num_sents_without_normal += 1
        elif "not" in sent:
            num_sents_without_normal += 1
    return NORMAL if num_sents_without_normal < 2 else ABNORMAL


@labeling_function()
def LF_normal(x):
    r = re.compile("No acute cardiopulmonary abnormality", re.IGNORECASE)
    for s in x.text.split("."):
        if r.search(s):
            return NORMAL
    return ABSTAIN


@labeling_function()
def LF_positive_MeshTerm(x):
    for idx in range(1, len(categories)):
        reg_pos = re.compile(categories[idx], re.IGNORECASE)
        reg_neg = re.compile(
            "(No|without|resolution)\\s([a-zA-Z0-9\-,_]*\\s){0,10}" + categories[idx],
            re.IGNORECASE,
        )
        for s in x.text.split("."):
            if (
                reg_pos.search(s)
                and (not reg_neg.search(s))
                and (not reg_equivocation.search(s))
            ):
                return ABNORMAL
    return ABSTAIN


@labeling_function()
def LF_fracture(x):
    reg_pos = re.compile("fracture", re.IGNORECASE)
    reg_neg = re.compile(
        "(No|without|resolution)\\s([a-zA-Z0-9\-,_]*\\s){0,10}fracture", re.IGNORECASE
    )
    for s in x.text.split("."):
        if (
            reg_pos.search(s)
            and (not reg_neg.search(s))
            and (not reg_equivocation.search(s))
        ):
            return ABNORMAL
    return ABSTAIN


@labeling_function()
def LF_calcinosis(x):
    reg_01 = re.compile("calc", re.IGNORECASE)
    reg_02 = re.compile("arter|aorta|muscle|tissue", re.IGNORECASE)
    for s in x.text.split("."):
        if reg_01.search(s) and reg_02.search(s):
            return ABNORMAL
    return ABSTAIN


@labeling_function()
def LF_degen_spine(x):
    reg_01 = re.compile("degen", re.IGNORECASE)
    reg_02 = re.compile("spine", re.IGNORECASE)
    for s in x.text.split("."):
        if reg_01.search(s) and reg_02.search(s):
            return ABNORMAL
    return ABSTAIN


@labeling_function()
def LF_lung_hypoinflation(x):
    reg_01 = re.compile(
        "hypoinflation|collapse|(low|decrease|diminish)\\s([a-zA-Z0-9\-,_]*\\s){0,4}volume",
        re.IGNORECASE,
    )
    for s in x.text.split("."):
        if reg_01.search(s):
            return ABNORMAL
    return ABSTAIN


@labeling_function()
def LF_lung_hyperdistention(x):
    reg_01 = re.compile("increased volume|hyperexpan|inflated", re.IGNORECASE)
    for s in x.text.split("."):
        if reg_01.search(s):
            return ABNORMAL
    return ABSTAIN


@labeling_function()
def LF_catheters(x):
    reg_01 = re.compile(" line|catheter|PICC", re.IGNORECASE)
    for s in x.text.split("."):
        if reg_01.search(s):
            return ABNORMAL
    return ABSTAIN


@labeling_function()
def LF_surgical(x):
    reg_01 = re.compile("clip", re.IGNORECASE)
    for s in x.text.split("."):
        if reg_01.search(s):
            return ABNORMAL
    return ABSTAIN


@labeling_function()
def LF_granuloma(x):
    reg_01 = re.compile("granuloma", re.IGNORECASE)
    for s in x.text.split("."):
        if reg_01.search(s):
            return ABNORMAL
    return ABSTAIN
