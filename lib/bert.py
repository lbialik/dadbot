from transformers import BertForMaskedLM
from transformers import BertModel
from transformers import BertTokenizer
import torch
from typing import List
from typing import Tuple


class ReRanker:
    PRETRAINED_MODEL_NAME = "bert-base-uncased"

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.PRETRAINED_MODEL_NAME)

        self.sentence_model = BertModel.from_pretrained(self.PRETRAINED_MODEL_NAME)
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

        scores = self.language_model(
            masked_sentence_ids, masked_lm_labels=masked_sentence_ids
        )[1][0, masked_index]

        distribution = torch.softmax(scores, 0)

        prediction = distribution[
            self.tokenizer.encode([sentence[masked_index]], add_special_tokens=False)[
                0
            ],
        ]

        return 1 - prediction


# class ReRanker:
#     """
#     Re-ranks candidate pun sentences based on local and global similarities
#     """

#     def __init__(self, model, masked_model):
#         self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#         self.model = model
#         self.masked_model = masked_model

#     def pre_process(self, sentences):
#         # process context sentence
#         sentence = "[CLS]"
#         # print(sentences)
#         for sent in sentences:
#             # print(sent)
#             sentence += " " + sent + " [SEP]"
#         tokenized_sentence = self.tokenizer.tokenize(sentence)
#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
#         return tokenized_sentence, indexed_tokens

#         # Convert inputs to PyTorch tensors
#         tokens_tensor = torch.tensor([indexed_tokens])
#         segments_tensors = torch.tensor([segments_ids])
#         return sentence

#     def find_surprisal(self, sentence, masked_index, segments_ids):
#         masked_sentence = sentence[:]
#         pun_word = sentence[masked_index]
#         masked_sentence[masked_index] = "[MASK]"
#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(masked_sentence)
#         tokens_tensor = torch.tensor([indexed_tokens])
#         segments_tensors = torch.tensor([segments_ids])

#         # Predict all tokens
#         with torch.no_grad():
#             outputs = self.masked_model(tokens_tensor, token_type_ids=segments_tensors)
#             predictions = outputs[0]
#         print(predictions)
#         # # confirm we were able to predict 'henson'
#         # predicted_index = torch.argmax(predictions[0, masked_index]).item()
#         # predicted_tokens = tokenizer.convert_ids_to_tokens([predicted_index])
#         surprisal = 0
#         return surprisal

#     def find_similarity(self, sen1, sen2):
#         indexed_tokens_1 = self.tokenizer.convert_tokens_to_ids(sen1)
#         indexed_tokens_2 = self.tokenizer.convert_tokens_to_ids(sen2)
#         segments_ids_1 = [1] * len(sen1)
#         segments_ids_2 = [1] * len(sen2)
#         # Convert inputs to PyTorch tensors
#         tokens_tensor_1 = torch.tensor([indexed_tokens_1])
#         tokens_tensor_2 = torch.tensor([indexed_tokens_1])
#         segments_tensors_1 = torch.tensor([segments_ids_1])
#         segments_tensors_2 = torch.tensor([segments_ids_2])

#         with torch.no_grad():
#             outputs_1, _ = self.model(
#                 tokens_tensor_1, token_type_ids=segments_tensors_1
#             )
#             print(len(outputs_1))
#             print("HEREEEEEEEEEe")
#             print(outputs_1)
#             print("HELLOOOOOOOOOOOO")
#             cls_1 = outputs_1
#             outputs_2, _ = self.model(
#                 tokens_tensor_2, token_type_ids=segments_tensors_2
#             )
#             print(len(outputs_2))
#             print(outputs_2)
#             cls_2 = outputs_2

#         print(cls_1, cls_2)
#         cos = torch.nn.CosineSimilarity()
#         diff = cos(cls_1, cls_2)
#         print(diff)
#         return diff

#     def rerank(self, original_sentence, context_sentence, potential_puns):
#         tokenized_original, original_sentence_tensors = self.pre_process(
#             [original_sentence]
#         )
#         tokenized_context, context_tensors = self.pre_process([context_sentence])
#         reranked_puns = []

#         # Put the model in "evaluation" mode, meaning feed-forward operation.
#         self.model.eval()

#         og_similarity = self.find_similarity(tokenized_original, tokenized_context)
#         for pun in potential_puns:
#             punned = [
#                 word for word in pun.split() if word not in original_sentence.split()
#             ][0]
#             index = pun.split().index(punned) + 1
#             tokenized_pun, pun_tensors = self.pre_process([pun])
#             tokenized_global, global_context_tensors = self.pre_process(
#                 [context_sentence, pun]
#             )
#             local_segments_id = [1] * len(tokenized_pun)
#             global_segments_id = [1] * (len(tokenized_pun)) + [2] * (
#                 len(tokenized_context)
#             )
#             local_surprisal = self.find_surprisal(
#                 tokenized_pun, index, local_segments_id
#             )
#             global_surprisal = self.find_surprisal(
#                 tokenized_global, index + len(tokenized_context), global_segments_id
#             )
#             pun_similarity = self.find_similarity(tokenized_pun, tokenized_context)
#             similarity_score = og_similarity - pun_similarity
#             surprisal_score = global_surprisal - local_surprisal
#             score = similarity_score + surprisal_score
#             print(original_sentence, " --> ", pun)
#             print(global_surprisal, local_surprisal)
#             print(og_similarity, pun_similarity)
#             print(score)
#             reranked_puns.append((pun, score))

#         return reranked_puns
