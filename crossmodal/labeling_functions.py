import re

# Defining labels
ABSTAIN = 0
ABNORMAL = 1
NORMAL= 2

def LF_report_is_short(report):
    """
    Checks if report is short.
    """
    return NORMAL if len(report) < 280 else ABSTAIN

negative_inflection_words = ["but", "however", "otherwise"]
def LF_negative_inflection_words_in_report(report):
    return ABNORMAL if any(word in report.lower() \
                      for word in negative_inflection_words) else ABSTAIN

def LF_is_seen_or_noted_in_report(report):
    return ABNORMAL if any(word in report.lower() \
                      for word in ["is seen", "noted"]) else ABSTAIN

def LF_disease_in_report(report):
    return ABNORMAL if "disease" in report.lower() else ABSTAIN

def LF_recommend_in_report(report):
    return ABNORMAL if "recommend" in report.lower() else ABSTAIN

def LF_mm_in_report(report):
    return ABNORMAL if any(word in report.lower() \
                      for word in ["mm", "cm"]) else ABSTAIN

abnormal_mesh_terms = ["opacity", "cardiomegaly", "calcinosis",
                       "hypoinflation", "calcified granuloma",
                       "thoracic vertebrae", "degenerative",
                       "hyperdistention", "catheters",
                       "granulomatous", "nodule", "fracture"
                       "surgical", "instruments", "emphysema"]
def LF_abnormal_mesh_terms_in_report(report):
    if any(mesh in report.lower() for mesh in abnormal_mesh_terms):
        return ABNORMAL
    else:
        return ABSTAIN

words_indicating_normalcy = ['clear', 'no', 'normal', 'unremarkable',
                             'free', 'midline']
def LF_consistency_in_report(report):
    '''
    The words 'clear', 'no', 'normal', 'free', 'midline' in
    findings section of the report
    '''
    report = report
    findings = report[report.find('FINDINGS:'):]
    findings = findings[:findings.find('IMPRESSION:')]
    sents = findings.split('.')

    num_sents_without_normal = ABSTAIN
    for sent in sents:
        sent = sent.lower()
        if not any(word in sent for word in words_indicating_normalcy):
            num_sents_without_normal += 1
        elif 'not' in sent:
            num_sents_without_normal += 1
    return NORMAL if num_sents_without_normal < 2 else ABNORMAL

categories = ['normal','opacity','cardiomegaly','calcinosis',
              'lung/hypoinflation','calcified granuloma',
              'thoracic vertebrae/degenerative','lung/hyperdistention',
              'spine/degenerative','catheters, indwelling',
              'granulomatous disease','nodule','surgical instruments',
              'scoliosis', 'osteophyte', 'spondylosis','fractures, bone']

def LF_normal(report):
    r = re.compile('No acute cardiopulmonary abnormality',re.IGNORECASE)
    for s in report.split("."):
        if r.search(s):
            return NORMAL
    return ABSTAIN

reg_equivocation = re.compile('unlikely|likely|suggests|questionable|concerning|possibly|potentially|could represent|may represent|may relate|cannot exclude|can\'t exclude|may be',re.IGNORECASE)

def LF_positive_MeshTerm(report):
    for idx in range(1,len(categories)):
        reg_pos = re.compile(categories[idx],re.IGNORECASE)
        reg_neg = re.compile('(No|without|resolution)\\s([a-zA-Z0-9\-,_]*\\s){0,10}'+categories[idx],re.IGNORECASE)
        for s in report.split("."):
            if reg_pos.search(s) and (not reg_neg.search(s)) and (not reg_equivocation.search(s)):
                return ABNORMAL
    return ABSTAIN

def LF_fracture(report):
    reg_pos = re.compile('fracture',re.IGNORECASE)
    reg_neg = re.compile('(No|without|resolution)\\s([a-zA-Z0-9\-,_]*\\s){0,10}fracture',re.IGNORECASE)
    for s in report.split("."):
        if reg_pos.search(s) and (not reg_neg.search(s)) and (not reg_equivocation.search(s)):
            return ABNORMAL
    return ABSTAIN

def LF_calcinosis(report):
    reg_01 = re.compile('calc',re.IGNORECASE)
    reg_02 = re.compile('arter|aorta|muscle|tissue',re.IGNORECASE)
    for s in report.split("."):
        if reg_01.search(s) and reg_02.search(s):
            return ABNORMAL
    return ABSTAIN

def LF_degen_spine(report):
    reg_01 = re.compile('degen',re.IGNORECASE)
    reg_02 = re.compile('spine',re.IGNORECASE)
    for s in report.split("."):
        if reg_01.search(s) and reg_02.search(s):
            return ABNORMAL
    return ABSTAIN

def LF_lung_hypoinflation(report):
    #reg_01 = re.compile('lung|pulmonary',re.IGNORECASE)
    reg_01 = re.compile('hypoinflation|collapse|(low|decrease|diminish)\\s([a-zA-Z0-9\-,_]*\\s){0,4}volume',re.IGNORECASE)
    for s in report.split("."):
        if reg_01.search(s):
            return ABNORMAL
    return ABSTAIN

def LF_lung_hyperdistention(report):
    #reg_01 = re.compile('lung|pulmonary',re.IGNORECASE)
    reg_01 = re.compile('increased volume|hyperexpan|inflated',re.IGNORECASE)
    for s in report.split("."):
        if reg_01.search(s):
            return ABNORMAL
    return ABSTAIN

def LF_catheters(report):
    reg_01 = re.compile(' line|catheter|PICC',re.IGNORECASE)
    for s in report.split("."):
        if reg_01.search(s):
            return ABNORMAL
    return ABSTAIN

def LF_surgical(report):
    reg_01 = re.compile('clip',re.IGNORECASE)
    for s in report.split("."):
        if reg_01.search(s):
            return ABNORMAL
    return ABSTAIN

def LF_granuloma(report):
    reg_01 = re.compile('granuloma',re.IGNORECASE)
    for s in report.split("."):
        if reg_01.search(s):
            return ABNORMAL
    return ABSTAIN