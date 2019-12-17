import sys
import traceback

import lib.puns as puns


def main(argv):
    punner = puns.Punner(config=puns.PunnerConfig())
    rerank = True
    while True:
        try:
            topic = input("Topic > ")
            if rerank:
                context = input("Please use it in a sentence > ")
            sentence = input("Sentence > ")

            pun, costs = punner.punnify(topic, sentence, context='')
            print(pun)
            print(costs)
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
    print()


if __name__ == "__main__":
    main(sys.argv)
