import math
import queue
import sys
from typing import Optional

import lib.pronunciation as pronunciation
import lib.semantics as semantics


class PunnerConfig:
    """
    Provides configuration for the Punner class. Lets us easily change
    hyperparameters when constructing the class.
    """

    DEFAULT_WORD_VECTOR_MODEL = semantics.ServerSimilarWordMap
    DEFAULT_SIMILAR_WORD_COUNT = 30
    DEFAULT_PHONOLOGY_WEIGHT = 1.0
    DEFAULT_SEMANTIC_WEIGHT = 1.0
    DEFAULT_REPLACE_COUNT = 1

    def __init__(self, **kwargs):
        # The word vector model we want to use.
        self.word_vector_model = kwargs.get(
            "word_vector_model", self.DEFAULT_WORD_VECTOR_MODEL
        )

        # The number of candidate words we consider around the topic word.
        self.similar_word_count = kwargs.get(
            "similar_word_count", self.DEFAULT_SIMILAR_WORD_COUNT
        )

        # The weight we give to phonological distance.
        self.phonology_weight = kwargs.get(
            "phonology_weight", self.DEFAULT_PHONOLOGY_WEIGHT
        )

        # The weight we give to semantic distance.
        self.semantic_weight = kwargs.get(
            "semantic_weigth", self.DEFAULT_SEMANTIC_WEIGHT
        )

        # The count of the sentence we replace with candidate words.
        self.replace_count = kwargs.get("replace_count", self.DEFAULT_REPLACE_COUNT)


class Punner:
    def __init__(self, config=None):
        self.config = config or PunnerConfig()
        self.word_vector_model = config.word_vector_model()

    def punnify(self, topic, sentence):
        """
        Given a topic and a sentence, produce a punnified version of the
        sentence where certain words have been replaced with those that are
        phonologically similar to the original word, and semantically similar
        to the topic word.
        """
        topic = self.tokenize(topic)[0]
        sentence = self.tokenize(sentence)
        candidate_words = self.word_vector_model.get_similar_words(
            topic, self.config.similar_word_count
        )

        # Calculates the best words to replace with for each position in the
        # sentence
        best_words = [("", sys.float_info.max)] * len(sentence)
        for i in range(len(sentence)):
            for (candidate_word, semantic_similarity) in candidate_words:
                phonology_cost = pronunciation.word_phonemic_distance(
                    # TODO: One of:
                    #   1) Search over all pronunciations
                    #   2) Find out if there's a pattern about American vs.
                    #      British, and always choose American.
                    pronunciation.word_to_phonemes(sentence[i])[0],
                    pronunciation.word_to_phonemes(candidate_word)[0],
                )
                semantic_cost = 1 - semantic_similarity

                cost = (phonology_cost * self.config.phonology_weight) + (
                    semantic_cost * self.config.semantic_weight
                )

                if cost < best_words[i][1]:
                    best_words[i] = (candidate_word, cost)

        # Calculating the actual indices that we should replace
        replace_count = self.config.replace_count
        replacements = sorted(enumerate(best_words), key=lambda group: group[1][1])[
            :replace_count
        ]

        # Replacing words in the sentence
        for (i, (best_word, _)) in replacements:
            sentence[i] = best_word

        return self.untokenize(sentence)

    def tokenize(self, string):
        """
        Strips all non-alphabetical characters (except for intra-word
        apostrophes), maps onto lowercase, and then splits the words by
        whitespace.
        """
        return string.lower().split()

    def untokenize(self, tokens):
        """
        Naively maps a list of tokens onto a well-formatted sentence. I.e. it
        capitalizes the first word and adds a period to the end.
        """
        return " ".join(tokens).capitalize() + "."
