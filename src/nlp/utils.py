import re
import configparser


def read_config(path_to_config: str) -> configparser.ConfigParser:
    """Method to parse .ini config file

    Args:
        path_to_config (str): path to .ini config file

    Returns:
        configparser.ConfigParser: parsed .ini config file
    """
    config = configparser.ConfigParser()
    config.read(path_to_config)
    return config


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
