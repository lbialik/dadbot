import gensim.parsing.preprocessing as preprocessing
import math
import queue
import sys
from typing import Optional
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

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
        self.threshold = 1.25

    def punnify(self, topic, sentence, context):
        """
        Given a topic and a sentence, produce a punnified version of the
        sentence where certain words have been replaced with those that are
        phonologically similar to the original word, and semantically similar
        to the topic word.
        """
        self.reranking = False
        if len(context) > 0:
            self.reranking = True
        topic = self.tokenize(topic)[0]
        sentence = self.tokenize(sentence)
        candidate_words = self.normalize_similarity_range(
            self.word_vector_model.get_similar_words(
                topic, self.config.similar_word_count
            )
        )

        # Calculates the best words to replace with for each position in the
        # sentence
        best_words = [("", sys.float_info.max)] * len(sentence)
        if self.reranking:
            best_words = [[("", sys.float_info.max)] for i in range(len(sentence))]
        for i in range(len(sentence)):
            # Skipping stopwords and words that have no pronunciation
            sentence_word_phonemes = pronunciation.word_to_phonemes(sentence[i])
            if (
                len(sentence_word_phonemes) == 0
                or sentence[i] in preprocessing.STOPWORDS
            ):
                continue
            sentence_word_phonemes = sentence_word_phonemes[0]

            for (candidate_word, semantic_similarity) in candidate_words:
                # Skipping stopwords, words that have no pronunciation, and
                # words that are equal to the word we already have.
                candidate_word_phonemes = pronunciation.word_to_phonemes(candidate_word)
                if (
                    len(candidate_word_phonemes) == 0
                    or candidate_word in preprocessing.STOPWORDS
                    or candidate_word == sentence[i]
                ):
                    continue
                candidate_word_phonemes = candidate_word_phonemes[0]

                phonology_cost = pronunciation.word_phonemic_distance(
                    # TODO: One of:
                    #   1) Search over all pronunciations
                    #   2) Find out if there's a pattern about American vs.
                    #      British, and always choose American.
                    sentence_word_phonemes,
                    candidate_word_phonemes,
                )
                semantic_cost = 1 - semantic_similarity

                cost = (phonology_cost * self.config.phonology_weight) + (
                    semantic_cost * self.config.semantic_weight
                )
                if self.reranking:
                    best_words[i].append((candidate_word, cost))
                else:
                    if cost < best_words[i][1]:
                        best_words[i] = (candidate_word, cost)
            if self.reranking:
                best_words[i].sort(key=lambda x: x[1])

        # Calculating the actual indices that we should replace
        # replace_count = self.config.replace_count
        # replacements = sorted(enumerate(best_words), key=lambda group: group[1][1])[
        #     :replace_count
        # ]
        replacements =  [replacement for replacement in sorted(enumerate(best_words), key=lambda group: group[1][1]) if replacement[1][1]<self.threshold]

        # Replacing words in the sentence
        costs = []
        for (i, (best_word, cost)) in replacements:
            costs.append((best_word, cost))
            sentence[i] = best_word

        return self.untokenize(sentence), costs

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

class ReRanker:
    """
    Re-ranks candidate pun sentences based on local and global similarities
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def rerank(self, original_sentence, context_sentence, potential_puns):
        # process context sentence
        context = context if context[-1] == '.' else context + '.'
        context = '[CLS] ' + context + ' [SEP]'
        tokenized_context = self.tokenizer.tokenize(context)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_context)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Load pre-trained model (weights)
        model = BertModel.from_pretrained('bert-base-uncased')

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()