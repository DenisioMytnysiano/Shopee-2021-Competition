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
    sentence = sentence.lower()
    sentence = sentence.replace(r"[^\w\s]", "")
    sentence = sentence.replace(r"\s+$", "")
    sentence = sentence.replace(r"x\d+$", "")
    sentence = sentence.replace(r"\W*\b\w{1,3}\b", "")
    return sentence
