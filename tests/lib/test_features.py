import cmudict
import re

from lib.features import features
from lib.pronunciation import *


def test_has_all_phonemes():
    """
    Tests that our feature set includes all of the possible symbols that CMU
    Dict can return.
    """
    missing_phonemes = set()
    for symbol in cmudict.symbols():
        no_stress = re.sub(r"\d+", "", symbol)
        if no_stress not in features:
            missing_phonemes.add(no_stress)

    # We expect to be missing exactly "AW", "AY", ER", "EY", "OW", and "OY".
    # Because they are all either diphthongs (which we map to monophthongs), or
    # for "ER" it's literally just the "R" sound.
    assert missing_phonemes == set(["AW", "AY", "ER", "EY", "OW", "OY"])


def test_phonemic_levenshtein():
    assert (
        word_phonemic_distance(word_to_phonemes("reed")[0], word_to_phonemes("read")[1])
        == 0
    )


def run_tests():
    test_has_all_phonemes()
    test_phonemic_levenshtein()
    print("All tests passed!")
