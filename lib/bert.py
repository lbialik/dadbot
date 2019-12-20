from transformers import BertForMaskedLM
from transformers import BertForNextSentencePrediction
from transformers import BertTokenizer
import torch
from typing import List
from typing import Tuple


class ReRanker:
    PRETRAINED_MODEL_NAME = "bert-base-uncased"

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.PRETRAINED_MODEL_NAME)

        self.sentence_model = BertForNextSentencePrediction.from_pretrained(
            self.PRETRAINED_MODEL_NAME
        )
        self.sentence_model.eval()

        self.language_model = BertForMaskedLM.from_pretrained(
            self.PRETRAINED_MODEL_NAME
        )
        self.language_model.eval()

    def rerank(
        self,
        context_sentence: List[str],
        potential_puns: List[Tuple[List[str], int, float]],
    ) -> List[Tuple[List[str], int, float]]:
        """
        Re-ranks a list of potential puns based on the original sentence and a
        provided context sentence. Based on Hehe et. al's work (2019) work as
        inspiration.

        We take global surprise to be the punned sentence followed by the
        context sentence, and we take local surprise to be the punned sentence
        alone.

        We maximize the ratio local/global, to simultaneously minimize global
        surprise and maximize local surprise.
        """
        # TODO: Vectorize
        local_surprises = [
            self._calculate_surprisal(
                ["[CLS]"] + potential_pun + ["[SEP]"],
                index + 1,  # We need to add 1 to the index to offset the CLS token
            )
            for (potential_pun, index, _) in potential_puns
        ]

        global_surprises = [
            self._calculate_surprisal(
                ["[CLS]"] + context_sentence + ["[SEP]"] + potential_pun + ["[SEP]"],
                index + 2 + len(context_sentence),
            )
            for (potential_pun, index, _) in potential_puns
        ]

        next_sentences = [
            self._calculate_next_sentence(context_sentence, potential_pun)
            for (potential_pun, index, _) in potential_puns
        ]

        ratios = [
            (local_surprises[i] / ((next_sentences[i] + global_surprises[i]) / 2), i)
            for i in range(len(potential_puns))
        ]
        ratios.sort(key=lambda x: x[0], reverse=True)

        reranked_puns = []
        for (_, i) in ratios:
            reranked_puns.append(potential_puns[i])
        return reranked_puns

    def _calculate_surprisal(self, sentence: List[str], masked_index: int) -> float:
        """
        Calculates the surprisal of a word appearing at a given index in the
        sentence. Uses BERT's raw masked language model to predict the
        probability distribution of the word at the masked index, and returns
        the inverse probability of the real word.
        """
        masked_sentence = sentence[:]
        masked_sentence[masked_index] = "[MASK]"

        masked_sentence_ids = torch.tensor(
            [self.tokenizer.encode(" ".join(masked_sentence), add_special_tokens=False)]
        )
        token_type_ids = self._get_token_type_ids(masked_sentence_ids)

        with torch.no_grad():
            scores = self.language_model(
                masked_sentence_ids,
                token_type_ids=token_type_ids,
                masked_lm_labels=masked_sentence_ids,
            )[1][0, masked_index]

        distribution = torch.softmax(scores, 0)

        prediction = distribution[
            # We have to reuse the BertTokenizer to give us the index of our
            # target word in the vocabulary.
            self.tokenizer.encode([sentence[masked_index]], add_special_tokens=False)[
                0
            ],
        ]

        return 1 - prediction

    def _calculate_next_sentence(
        self, antecedent_sentence: List[str], subsequent_sentence: List[str]
    ) -> float:
        """
        Calculates the probability that the subsequent sentence follows the
        antecedent sentence.
        """
        sentence = (
            ["[CLS]"]
            + antecedent_sentence
            + ["[SEP]"]
            + subsequent_sentence
            + ["[SEP]"]
        )

        sentence_ids = torch.tensor(
            [self.tokenizer.encode(" ".join(sentence), add_special_tokens=False)]
        )
        token_type_ids = self._get_token_type_ids(sentence_ids)

        with torch.no_grad():
            scores = self.sentence_model(sentence_ids, token_type_ids=token_type_ids)[
                0
            ][0]

        distribution = torch.softmax(scores, 0)

        return distribution[1]

    def _get_token_type_ids(self, sentence_ids: List[int]) -> List[int]:
        """
        From a list of tokens formatted for BERT, i.e. includes [CLS] and
        [SEP], produce the token_type_ids necessary for the model.
        """
        sep_id = self.tokenizer.encode("[SEP]", add_special_tokens=False)[0]

        token_type_ids = []
        index = 0
        for id in sentence_ids[0]:
            token_type_ids.append(index)
            if id == sep_id:
                index += 1
        return torch.tensor(token_type_ids)
