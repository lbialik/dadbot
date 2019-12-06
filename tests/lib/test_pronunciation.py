import lib.pronunciation as pronunciation


def test_expand_phoneme():
    """
    Tests that we correctly expand diphthongs into the right vowel pairs. In
    the case of "ER" we just map it onto "R" because they're the same sound.
    """

    assert ["AE", "UH"] == pronunciation.expand_phoneme("AW")
    assert ["AE", "IH"] == pronunciation.expand_phoneme("AY")
    assert ["R"] == pronunciation.expand_phoneme("ER")
    assert ["E", "IH"] == pronunciation.expand_phoneme("EY")
    assert ["O", "UH"] == pronunciation.expand_phoneme("OW")
    assert ["AO", "IH"] == pronunciation.expand_phoneme("OY")


def test_word_to_phonemes():
    """
    Enumerates all of the example words on the CMUdict website, and ensures
    that we have the correct pronunciation.
    """

    assert [["AO", "T"]] == pronunciation.word_to_phonemes("ought")
    assert [["K", "AE", "UH"]] == pronunciation.word_to_phonemes("cow")
    assert [["HH", "AE", "IH", "D"]] == pronunciation.word_to_phonemes("hide")
    assert [["B", "IY"]] == pronunciation.word_to_phonemes("be")[
        :1
    ]  # contains an extra item
    assert [["CH", "IY", "Z"]] == pronunciation.word_to_phonemes("cheese")
    assert [["D", "IY"]] == pronunciation.word_to_phonemes("dee")
    assert [["DH", "IY"]] == pronunciation.word_to_phonemes("thee")
    assert [["EH", "D"]] == pronunciation.word_to_phonemes("ed")
    assert [["HH", "R", "T"]] == pronunciation.word_to_phonemes("hurt")
    assert [["E", "IH", "T"]] == pronunciation.word_to_phonemes("ate")
    assert [["F", "IY"]] == pronunciation.word_to_phonemes("fee")
    assert [["G", "R", "IY", "N"]] == pronunciation.word_to_phonemes("green")
    assert [["HH", "IY"]] == pronunciation.word_to_phonemes("he")
    assert [["IH", "T"]] == pronunciation.word_to_phonemes("it")[
        :1
    ]  # contains an extra item
    assert [["IY", "T"]] == pronunciation.word_to_phonemes("eat")
    assert [["JH", "IY"]] == pronunciation.word_to_phonemes("gee")
    assert [["K", "IY"]] == pronunciation.word_to_phonemes("key")
    assert [["L", "IY"]] == pronunciation.word_to_phonemes("lee")
    assert [["M", "IY"]] == pronunciation.word_to_phonemes("me")
    assert [["N", "IY"]] == pronunciation.word_to_phonemes("knee")
    assert [["P", "IH", "NG"]] == pronunciation.word_to_phonemes("ping")
    assert [["O", "UH", "T"]] == pronunciation.word_to_phonemes("oat")
    assert [["T", "AO", "IH"]] == pronunciation.word_to_phonemes("toy")
    assert [["P", "IY"]] == pronunciation.word_to_phonemes("pee")
    assert [["R", "EH", "D"], ["R", "IY", "D"]] == pronunciation.word_to_phonemes(
        "read"
    )
    assert [["S", "IY"]] == pronunciation.word_to_phonemes("sea")
    assert [["SH", "IY"]] == pronunciation.word_to_phonemes("she")
    assert [["T", "IY"]] == pronunciation.word_to_phonemes("tea")
    assert [["TH", "E", "IH", "T", "AH"]] == pronunciation.word_to_phonemes("theta")
    assert [["HH", "UH", "D"]] == pronunciation.word_to_phonemes("hood")
    assert [["T", "UW"]] == pronunciation.word_to_phonemes("two")
    assert [["V", "IY"]] == pronunciation.word_to_phonemes("vee")
    assert [["W", "IY"]] == pronunciation.word_to_phonemes("we")
    assert [["Y", "IY", "L", "D"]] == pronunciation.word_to_phonemes("yield")
    assert [["Z", "IY"]] == pronunciation.word_to_phonemes("zee")
    assert [["S", "IY", "ZH", "R"]] == pronunciation.word_to_phonemes("seizure")


def test_word_phonemic_distance_trivial():
    """
    Tests that word_phonemic_distance functions correctly when there is
    nothing to change.
    """

    assert 0 == pronunciation.word_phonemic_distance(
        pronunciation.word_to_phonemes("reed")[0],
        pronunciation.word_to_phonemes("read")[1],
    )


def test_word_phonemic_distance_insertion():
    """
    Tests that, when we only perform insertion, we produce the correct value.
    """
    assert 1 == pronunciation.word_phonemic_distance(
        pronunciation.word_to_phonemes("it")[0],
        pronunciation.word_to_phonemes("its")[0],
        verbose=True,
    )

    assert 2 == pronunciation.word_phonemic_distance(
        [], pronunciation.word_to_phonemes("it")[0]
    )


def test_word_phonemic_distance_deletion():
    """
    Tests that, when we only perform deletion, we produce the correct value.
    """
    assert 1 == pronunciation.word_phonemic_distance(
        pronunciation.word_to_phonemes("its")[0],
        pronunciation.word_to_phonemes("it")[0],
    )

    assert 2 == pronunciation.word_phonemic_distance(
        pronunciation.word_to_phonemes("it")[0], []
    )
