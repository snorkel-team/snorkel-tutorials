# +
from typing import Optional

from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint


# -


@preprocessor()
def get_person_text(cand: DataPoint) -> DataPoint:
    """
    Returns the text for the two person mentions in candidate
    """
    person_names = []
    for index in [1, 2]:
        field_name = "person{}_word_idx".format(index)
        start = cand[field_name][0]
        end = cand[field_name][1] + 1
        person_names.append(" ".join(cand["tokens"][start:end]))
    cand.person_names = person_names
    return cand


@preprocessor()
def get_person_last_names(cand: DataPoint) -> DataPoint:
    """
    Returns the last names for the two person mentions in candidate
    """
    cand = get_person_text(cand)
    person1_name, person2_name = cand.person_names
    person1_lastname = (
        person1_name.split(" ")[-1] if len(person1_name.split(" ")) > 1 else None
    )
    person2_lastname = (
        person2_name.split(" ")[-1] if len(person2_name.split(" ")) > 1 else None
    )
    cand.person_lastnames = [person1_lastname, person2_lastname]
    return cand


@preprocessor()
def get_text_between(cand: DataPoint) -> DataPoint:
    """
    Returns the text between the two person mentions in the sentence
    """
    start = cand.person1_word_idx[1] + 1
    end = cand.person2_word_idx[0]
    cand.text_between = " ".join(cand.tokens[start:end])
    return cand


@preprocessor()
def get_left_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the length 3 window to the left of the person mentions
    """
    # TODO: need to pass window as input params
    window = 3

    end = cand.person1_word_idx[0]
    cand.person1_left_tokens = cand.tokens[0:end][-1 - window : -1]

    end = cand.person2_word_idx[0]
    cand.person2_left_tokens = cand.tokens[0:end][-1 - window : -1]
    return cand


# Helper function to get last name for dbpedia entries.
def last_name(s: str) -> Optional[str]:
    name_parts = s.split(" ")
    return name_parts[-1] if len(name_parts) > 1 else None
