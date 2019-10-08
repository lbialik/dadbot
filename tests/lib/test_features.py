import cmudict
import re

from lib.features import features


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
    assert missing_phonemes == set()
