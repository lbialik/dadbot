import cmudict
from functools import reduce
import re
import numpy as np

from lib.features import features, feature_weights


# We store the cmudict as an object in memory so that we don't have to reload
# it every single time we call word_to_phonemes.
cmudict_cache = cmudict.dict()


# Maps from diphthongs to their monophthonic parts. Also "ER" for no reason.
diphthong_pairs = {
    "AW": ["AE", "UH"],
    "AY": ["AE", "IH"],
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


def word_phonemic_distance(source="", target="", verbose=False):
    size_x = len(source) + 1
    size_y = len(target) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            matrix[x, y] = min(
                matrix[x - 1, y] + ins_cost(target[y - 1]),
                matrix[x - 1, y - 1] + sub_cost(source[x - 1], target[y - 1]),
                matrix[x, y - 1] + del_cost(source[x - 1]),
            )
    if verbose:
        print(matrix)
    return matrix[size_x - 1, size_y - 1]


def phonemic_distance(phon1, phon2):
    """
    Takes two phonemes and returns their distance 0-1
    """
    values = {"-": -1, "0": 0, "+": 1}
    dist = 0
    feats1 = features[phon1]
    feats2 = features[phon2]
    total_weight = 0
    for f1 in feature_weights:
        dist += feature_weights[f1] * abs(values[feats1[f1]] - values[feats2[f1]])
        total_weight += feature_weights[f1]
    return dist / (float(total_weight))


def ins_cost(phon):
    """
    Returns cost of inserting a given phoneme and index
    """
    if features[phon]["syllabic"] == "+":
        return 1.5
    return 0.75


def del_cost(phon):
    """
    Returns cost of deleting a given phoneme and index
    """
    if features[phon]["syllabic"] == "+":
        return 1.5
    return 0.75


def sub_cost(phon1, phon2):
    """
    Returns cost of replacing a given phoneme with another at given indices
    """
    if phon1 == phon2:
        return 0
    else:
        return phonemic_distance(phon1, phon2)
