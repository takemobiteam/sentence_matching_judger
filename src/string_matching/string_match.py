import re
import string
from nltk.stem import PorterStemmer
from pandas.core.series import Series


def remove_punctuation(text):
    """Remove punctuation from the input text."""
    # Create a translation table that maps punctuation characters to None
    translator = str.maketrans("", "", string.punctuation)

    # Use the translate method to remove punctuation
    return text.translate(translator)


def stem_sentence(sentence, stemmer):
    """apply nltk stemmer to sentence"""
    stemmed_words = [
        stemmer.stem(remove_punctuation(word.lower())) for word in sentence.split()
    ]
    stemmed_sentence = " ".join(stemmed_words)
    return stemmed_sentence


def is_word_in_text(substring, full_text):
    """Use regular expression to match the substring as a whole word
    \b indicates word boundaries (beginning, end, or surrounded by spaces)"""
    pattern = r"\b" + re.escape(substring) + r"\b"
    return bool(re.search(pattern, full_text))


def find_sentence_in_text(key_word, text, ps):
    """apply stemmer to each word in a sentence, then do a string match"""
    if text:
        sentences = re.split(r"\.|<br>", str(text))
        for sentence in sentences:
            if is_word_in_text(
                stem_sentence(key_word, ps), stem_sentence(sentence, ps)
            ):
                return sentence
    return None


def find_match(key_word: str, row: Series):
    """Given a row in a dataframe which contains all saved information
       for a matched (poi, tag) pair. do the string match again. I'm doing
       more preprocess here so there must be difference on the matching
       result, but it doesn't matter

    Args:
        key_word (str): keyword for matching, usualy it's the tag name
        row (pandas.core.series.Series): a row in a dataframe which contains all saved information
                      for a matched (poi, tag) pair

    Returns:
        field_key: where the matched sentence lies, can be <name, description or review>
        matched_sentence: matched sentence
    """
    ps = PorterStemmer()
    field_keys = ["name", "description", "review0", "review1", "review2"]
    for field_key in field_keys:
        matched_sentence = find_sentence_in_text(key_word, row[field_key], ps)
        if matched_sentence:
            return field_key, matched_sentence
    return "", ""
