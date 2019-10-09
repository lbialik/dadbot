import cmudict
from functools import reduce
import re

from lib.features import features


# We store the cmudict as an object in memory so that we don't have to reload
# it every single time we call word_to_phonemes.
cmudict_cache = cmudict.dict()


# Maps from diphthongs to their monophthonic parts. Also "ER" for no reason.
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
    return diphthong_pairs.get(phoneme, [phoneme])


def word_to_phonemes(word):
    """
    Maps a single word onto a series of possible pronunciation. Each
    pronunciation is a series of phonemes, represented in the format provided
    by CMU Dict. See http://www.speech.cs.cmu.edu/cgi-bin/cmudict for
    reference.
    """
    source_pronunciations = cmudict_cache[word]
    pronunciations = []

    for source_pronunciation in source_pronunciations:
        pronunciation = []
        for phoneme in source_pronunciation:
            pronunciation += expand_phoneme(re.sub(r"\d+", "", phoneme))
        pronunciations.append(pronunciation)

    return pronunciations


def word_to_feature_matrix(word):
    """
    Maps a single word onto a series of feature matrices, one for each
    pronunciation.
    """
    pronunciations = word_to_phonemes(word)
    feature_matrices = []

    for pronunciation in pronunciations:
        feature_matrix = []
        for phoneme in pronunciation:
            feature_matrix.append(features[phoneme])
        feature_matrices.append(feature_matrix)

    return feature_matrices
