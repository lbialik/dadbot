import cmudict
from functools import reduce
import re
from typing import Dict
from typing import List

from lib.features import features


# We store the cmudict as an object in memory so that we don't have to reload
# it every single time we call word_to_phonemes.
_cmudict_cache = cmudict.dict()


# Construct mappings from diphthongs to
_diphthong_pairs = {
    "AW": ["AE", "UH"],
    "AY": ["AA", "IH"],
    "ER": ["R"],
    "EY": ["E", "IH"],
    "OW": ["O", "UH"],
    "OY": ["AO", "IH"],
}


def _expand_phoneme(phoneme: str) -> List[str]:
    """
    Expands a phoneme to potentially multiple phonemes. Used to map diphthongs
    to its monophthongs in series.
    """
    return _diphthong_pairs.get(phoneme, [phoneme])


def word_to_phonemes(word: str) -> List[List[str]]:
    """
    Maps a single word onto a series of possible pronunciation. Each
    pronunciation is a series of phonemes, represented in the format provided
    by CMU Dict. See http://www.speech.cs.cmu.edu/cgi-bin/cmudict for reference
    """
    return [
        reduce(
            lambda x, y: x + y,
            [
                _expand_phoneme(
                    re.sub(r"\d+", "", phoneme)
                )  # Stripping stress markers from the words
                for phoneme in pronunciation
            ],
        )
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
