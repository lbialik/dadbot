import gensim.parsing.preprocessing as preprocessing
import math
import queue
import sys
import torch
from typing import Optional
from typing import List
from typing import Tuple

import lib.pronunciation as pronunciation
import lib.semantics as semantics


class PunnerConfig:
    """
    Provides configuration for the Punner class. Lets us easily change
    hyperparameters when constructing the class.
    """

    DEFAULT_WORD_VECTOR_MODEL = semantics.ServerSimilarWordMap
    DEFAULT_SIMILAR_WORD_COUNT = 200
    DEFAULT_PHONOLOGY_WEIGHT = 1.0
    DEFAULT_SEMANTIC_WEIGHT = 1.0
    DEFAULT_REPLACE_COUNT = 1
    DEFAULT_THRESHOLD = 1.2
    DEFAULT_RERANK_THRESHOLD = 2.3

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

        # The threshold by which we'll replace a word in a sentence
        self.threshold = kwargs.get("threshold", self.DEFAULT_THRESHOLD)

        # The threshold by which we'll replace word when we're reranking puns
        # with DadBERT.
        self.rerank_threshold = kwargs.get(
            "rerank_threshold", self.DEFAULT_RERANK_THRESHOLD
        )


class Punner:
    def __init__(self, config=None):
        self.config = config or PunnerConfig()
        self.word_vector_model = config.word_vector_model()

    def punnify(self, topic, sentence, context, model, masked_model):
        """
        Given a topic and a sentence, produce a punnified version of the
        sentence where certain words have been replaced with those that are
        phonologically similar to the original word, and semantically similar
        to the topic word.
        """
        reranking = False
        if len(context) > 0:
            reranking = True

        topic = self.tokenize(topic)[0]
        sentence = self.tokenize(sentence)
        candidate_words = self.normalize_similarity_range(
            self.word_vector_model.get_similar_words(
                topic, self.config.similar_word_count
            )
        )

        best_replacements = self._calculate_best_replacements(
            topic,
            sentence,
            context,
            candidate_words,
            self.config.threshold if not reranking else self.config.rerank_threshold,
        )

        if reranking:
            pun_candidates = self._generate_pun_candidates(sentence, best_replacements)
            # TODO: Fix and use ReRanker
        else:
            return self._generate_best_pun(sentence, best_replacements)

    def _calculate_best_replacements(
        self,
        topic: str,
        sentence: List[str],
        candidate_words: List[Tuple[str, float]],
        threshold: float,
    ) -> List[List[Tuple[str, float]]]:
        """
        Calculates the most optimal candidate words with which we can replace
        words in the sentence.

        The list we return is the length of the sentence. The list at index i
        corresponds to the best candidates for replacement at that index.
        """
        best_replacements = []
        for (i, sentence_word) in enumerate(sentence):
            best_replacements.append([])
            sentence_word_phonemes = self._get_word_phonemes(sentence_word)
            if sentence_word_phonemes is None:
                continue

            for (candidate_word, semantic_similarity) in candidate_words:
                candidate_word_phonemes = self._get_word_phonemes(candidate_word)
                if candidate_word_phonemes is None:
                    continue

                phonology_cost = pronunciation.word_phonemic_distance(
                    sentence_word_phonemes, candidate_word_phonemes
                )
                semantic_cost = 1 - semantic_similarity
                cost = (phonology_cost * self.config.phonology_weight) + (
                    semantic_cost * self.config.semantic_weight
                )

                if cost <= threshold:
                    best_replacements[i].append((candidate_word, cost))
            best_replacements[i].sort(key=lambda tuple: tuple[1])

        return best_replacements

    def _get_word_phonemes(self, word: str) -> Optional[str]:
        """
        Tries to retrieve the phonemic representation of a word. If the word
        doesn't have a pronunciation or if it is a stop-word, we just return
        None.
        """
        word_phonemes = pronunciation.word_to_phonemes(word)
        if len(word_phonemes) == 0 or word in preprocessing.STOPWORDS:
            return None
        return word_phonemes[0]

    def _generate_pun_candidates(
        self, sentence: List[str], best_replacements: List[List[Tuple[str, float]]]
    ) -> List[Tuple[List[str], float]]:
        """
        Generates each possible pun from a set of replacements that are
        sufficiently good.

        Assumes that len(sentence) == len(best_replacements)
        """
        punned_sentences = []
        for i in range(len(sentence)):
            for j in range(len(best_replacements)):
                punned_sentence = sentence[:]
                punned_sentence[i] = best_replacements[i][j][0]

                punned_sentences.append((punned_sentence, best_replacements[i][j][1]))

    def _generate_best_pun(
        self, sentence: List[str], best_replacements: List[List[Tuple[str, float]]]
    ) -> Tuple[List[str], List[float]]:
        """
        Generates one single best sentence. Replaces each word in the sentence
        with the best candidate replacement sentence at each index.

        Assumes that:
          1) len(sentence) == len(best_replacements)
          2) best_replacements is sorted, in descending order
        """
        punned_sentence = sentence[:]
        replacement_costs = []
        for i in range(len(sentence)):
            if len(best_replacements[i]) > 0:
                punned_sentence[i] = best_replacements[i][0][0]
                replacement_costs.append(best_replacements[i][0][1])
            else:
                replacement_costs.append(0)
        return punned_sentence, replacement_costs

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

    def normalize_similarity_range(self, candidate_words):
        """
        Given a list of candidate words, normalize the similarity metric
        between them from whatever their range is to [0, 1].
        """
        min_sim, max_sim = self.get_similarity_range(candidate_words)
        return [
            (candidate_word, (sim - min_sim) / (max_sim - min_sim))
            for (candidate_word, sim) in candidate_words
        ]

    def get_similarity_range(self, candidate_words):
        """
        Gets the maximum and minimum values of similarity in a list of
        candidate words.
        """
        min_sim = 1
        max_sim = 0
        for (_, sim) in candidate_words:
            if sim < min_sim:
                min_sim = sim
            if sim > max_sim:
                max_sim = sim
        return (min_sim, max_sim)
