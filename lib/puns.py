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
        replacements = []
        if self.reranking:
            potential_puns = []
            for (index, potential_replacements) in enumerate(best_words):
                words_to_consider = [
                    replacement
                    for replacement in potential_replacements
                    if replacement[1] < self.config.rerank_threshold
                ]
                for candidate_word in words_to_consider:
                    pun_sentence = sentence[:]
                    pun_sentence[index] = candidate_word[0]
                    potential_puns.append((pun_sentence, candidate_word[1]))
            # sentence, costs = potential_puns[0]
            RR = ReRanker(model, masked_model)
            potential_sentences = [self.untokenize(pun[0]) for pun in potential_puns]
            reranked_puns = RR.rerank(
                self.untokenize(sentence),
                self.untokenize(self.tokenize(context)),
                potential_sentences,
            )
            # print(reranked_puns)
            sentence, cost = reranked_puns[0]
        else:
            replacements = [
                replacement
                for replacement in sorted(
                    enumerate(best_words), key=lambda group: group[1][1]
                )
                if replacement[1][1] < self.config.threshold
            ]
            # Replacing words in the sentence
            cost = []
            for (i, (best_word, cost)) in replacements:
                cost.append((best_word, cost))
                sentence[i] = best_word

        return self.untokenize(sentence), cost

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

    def __init__(self, model, masked_model):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = model
        self.masked_model = masked_model

    def pre_process(self, sentences):
        # process context sentence
        sentence = "[CLS]"
        # print(sentences)
        for sent in sentences:
            # print(sent)
            sentence += " " + sent + " [SEP]"
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        return tokenized_sentence, indexed_tokens

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        return sentence

    def find_surprisal(self, sentence, masked_index, segments_ids):
        masked_sentence = sentence[:]
        pun_word = sentence[masked_index]
        masked_sentence[masked_index] = "[MASK]"
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(masked_sentence)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Predict all tokens
        with torch.no_grad():
            outputs = self.masked_model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]
        print(predictions)
        # # confirm we were able to predict 'henson'
        # predicted_index = torch.argmax(predictions[0, masked_index]).item()
        # predicted_tokens = tokenizer.convert_ids_to_tokens([predicted_index])
        surprisal = 0
        return surprisal

    def find_similarity(self, sen1, sen2):
        indexed_tokens_1 = self.tokenizer.convert_tokens_to_ids(sen1)
        indexed_tokens_2 = self.tokenizer.convert_tokens_to_ids(sen2)
        segments_ids_1 = [1] * len(sen1)
        segments_ids_2 = [1] * len(sen2)
        # Convert inputs to PyTorch tensors
        tokens_tensor_1 = torch.tensor([indexed_tokens_1])
        tokens_tensor_2 = torch.tensor([indexed_tokens_1])
        segments_tensors_1 = torch.tensor([segments_ids_1])
        segments_tensors_2 = torch.tensor([segments_ids_2])

        with torch.no_grad():
            outputs_1, _ = self.model(
                tokens_tensor_1, token_type_ids=segments_tensors_1
            )
            print(len(outputs_1))
            print("HEREEEEEEEEEe")
            print(outputs_1)
            print("HELLOOOOOOOOOOOO")
            cls_1 = outputs_1
            outputs_2, _ = self.model(
                tokens_tensor_2, token_type_ids=segments_tensors_2
            )
            print(len(outputs_2))
            print(outputs_2)
            cls_2 = outputs_2

        print(cls_1, cls_2)
        cos = torch.nn.CosineSimilarity()
        diff = cos(cls_1, cls_2)
        print(diff)
        return diff

    def rerank(self, original_sentence, context_sentence, potential_puns):
        tokenized_original, original_sentence_tensors = self.pre_process(
            [original_sentence]
        )
        tokenized_context, context_tensors = self.pre_process([context_sentence])
        reranked_puns = []

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

        og_similarity = self.find_similarity(tokenized_original, tokenized_context)
        for pun in potential_puns:
            punned = [
                word for word in pun.split() if word not in original_sentence.split()
            ][0]
            index = pun.split().index(punned) + 1
            tokenized_pun, pun_tensors = self.pre_process([pun])
            tokenized_global, global_context_tensors = self.pre_process(
                [context_sentence, pun]
            )
            local_segments_id = [1] * len(tokenized_pun)
            global_segments_id = [1] * (len(tokenized_pun)) + [2] * (
                len(tokenized_context)
            )
            local_surprisal = self.find_surprisal(
                tokenized_pun, index, local_segments_id
            )
            global_surprisal = self.find_surprisal(
                tokenized_global, index + len(tokenized_context), global_segments_id
            )
            pun_similarity = self.find_similarity(tokenized_pun, tokenized_context)
            similarity_score = og_similarity - pun_similarity
            surprisal_score = global_surprisal - local_surprisal
            score = similarity_score + surprisal_score
            print(original_sentence, " --> ", pun)
            print(global_surprisal, local_surprisal)
            print(og_similarity, pun_similarity)
            print(score)
            reranked_puns.append((pun, score))

        return reranked_puns
