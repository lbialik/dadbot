import sys
import torch
import traceback
from typing import List

import lib.bert as bert
import lib.puns as puns


def main(argv):
    reranker = None
    if len(argv) >= 2 and argv[1] == "rerank":
        reranker = bert.ReRanker()

    punner = puns.Punner(config=puns.PunnerConfig())
    while True:
        try:
            topic = input("Topic > ")
            context = ""
            if reranker is not None:
                context = input("Please use it in a sentence > ")
            sentence = input("Sentence > ")
            pun, costs = punner.punnify(topic, sentence, context, reranker)
            print(pun)
            print(costs)
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
    print()


if __name__ == "__main__":
    main(sys.argv)
