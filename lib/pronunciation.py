import cmudict
import re
from typing import Dict
from typing import List

from lib.features import features

_cmudict_cache = cmudict.dict()


def word_to_phonemes(word: str) -> List[List[str]]:
    """
    Maps a single word onto a series of possible pronunciation. Each
    pronunciation is a series of phonemes, represented in the format provided
    by CMU Dict. See http://www.speech.cs.cmu.edu/cgi-bin/cmudict for reference
    """
    return [
        [
            re.sub(r"\d+", "", phoneme)  # Stripping stress markers from the words
            for phoneme in pronunciation
        ]
        for pronunciation in _cmudict_cache[word]
    ]


def word_to_feature_matrix(word: str) -> List[List[Dict[str, str]]]:
    """
    Maps a single word onto a series of feature matrices, one for each
    pronunciation.
    """
    pronunciations = word_to_phonemes(word)
    return [
        [features[phoneme] for phoneme in pronunciation]
        for pronunciation in pronunciations
    ]
