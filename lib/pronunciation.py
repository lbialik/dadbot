import cmudict
from functools import reduce
import re

from lib.features import features


# We store the cmudict as an object in memory so that we don't have to reload
# it every single time we call word_to_phonemes.
cmudict_cache = cmudict.dict()


# Construct mappings from diphthongs to
diphthong_pairs = {
    "AW": ["AE", "UH"],
    "AY": ["AA", "IH"],
    "ER": ["R"],
    "EY": ["E", "IH"],
    "OW": ["O", "UH"],
    "OY": ["AO", "IH"],
}


def expand_phoneme(phoneme):
    """
    Expands a phoneme to potentially multiple phonemes. Used to map diphthongs
    to its monophthongs in series.
    """
    return _diphthong_pairs.get(phoneme, [phoneme])


def word_to_phonemes(word):
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
                    # Stripping stress markers from words
                    re.sub(r"\d+", "", phoneme)
                )
                for phoneme in pronunciation
            ],
        )
        for pronunciation in _cmudict_cache[word]
    ]


def word_to_feature_matrix(word):
    """
    Maps a single word onto a series of feature matrices, one for each
    pronunciation.
    """
    pronunciations = word_to_phonemes(word)
    return [
        [features[phoneme] for phoneme in pronunciation]
        for pronunciation in pronunciations
    ]
