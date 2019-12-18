import sys
import traceback

import lib.puns as puns
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

def main(argv):
    punner = puns.Punner(config=puns.PunnerConfig())
    rerank = True
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    masked_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    while True:
        try:
            topic = input("Topic > ")
            context = ''
            if rerank:
                context = input("Please use it in a sentence > ")
            sentence = input("Sentence > ")
            pun, costs = punner.punnify(topic, sentence, context, model, masked_model)
            print(pun)
            print(costs)
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
    print()


if __name__ == "__main__":
    main(sys.argv)
