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

        # TODO: Factor in sentence similarity

        ratios = [
            (local_surprises[i] / global_surprises[i], i)
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

        with torch.no_grad():
            scores = self.language_model(
                masked_sentence_ids,
                token_type_ids=self._get_token_type_ids(masked_sentence),
                masked_lm_labels=masked_sentence_ids,
            )[1][0, masked_index]

        distribution = torch.softmax(scores, 0)

        prediction = distribution[
            self.tokenizer.encode([sentence[masked_index]], add_special_tokens=False)[
                0
            ],
        ]

        return 1 - prediction

    def _get_token_type_ids(self, sentence: List[str]) -> List[int]:
        """
        From a list of tokens formatted for BERT, i.e. includes [CLS] and
        [SEP], produce the token_type_ids necessary for the model.
        """

        token_type_ids = []
        index = 0
        for word in sentence:
            token_type_ids.append(index)
            if word == "[SEP]":
                index += 1
        return token_type_ids
