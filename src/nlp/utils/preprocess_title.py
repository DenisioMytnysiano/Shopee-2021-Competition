import re


def preprocess_title(sentence: str) -> str:
    """Method to preprocess title:
        - lowercase the title
        - remove trailing spaces
        - remove encoding errors
        - remove punctuation
        - remove metrics (cm, mm, ml etc)

    Args:
        sentence (str): title string

    Returns:
        str: preprocessed title
    """
    result = sentence.lower()
    result = re.sub(r"[^\w\s]", "", result)
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"x\d+", "", result)
    result = re.sub(r"\W*\b\w{1,3}\b", "", result)
    return result
