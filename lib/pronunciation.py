import cmudict
from functools import reduce
import re

from features import features


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


def word_phonemic_distance(source, target):
    """
    Takes two lists of phonemes, returns a distance score between two words
    based on Levenshtein distance, using feature matrices for weights.
    """
    m = len(source)
    n = len(target)

    dist = [[0] * (m + 1) for i in range(n + 1)]

    for i in range(n + 1):
        for j in range(m + 1):
            d_cost = float(del_cost(source[j - 1], j - 1))
            i_cost = float(ins_cost(target[i - 1], i - 1))
            s_cost = float(sub_cost(source[j - 1], target[i - 1], j - 1, i - 1))
            i_cost = (
                i_cost * 2 * (float(i) / n)
            )  # sounds at the end are given more weight
            d_cost = d_cost * 2 * (float(j) / m)
            s_cost = s_cost * 2 * (float(j + i) / (m + n))
            dist[i][j] = min(
                dist[i - 1][j] + i_cost,
                dist[i][j - 1] + d_cost,
                dist[i - 1][j - 1] + s_cost,
            )
    return dist[i][j]


def phonemic_distance(phon1, phon2):
    """
    Takes two phonemes and returns their distance 0-1
    """
    values = {"-": -1, "0": 0, "+": 1}
    dist = 0
    feats1 = features[phon1]
    feats2 = features[phon2]
    for f1 in feats1:
        dist += abs(values[feats1[f1]] - values[feats2[f1]])
    return dist / (float(len(list(feats1))))


def ins_cost(phon, idx):
    """
    Returns cost of inserting a given phoneme and index
    """
    return 1


def del_cost(phon, idx):
    """
    Returns cost of deleting a given phoneme and index
    """
    return 1


def sub_cost(phon1, phon2, idx1, idx2):
    """
    Returns cost of replacing a given phoneme with another at given indices
    """
    if phon1 == phon2:
        return 0
    else:
        return phonemic_distance(phon1, phon2)
