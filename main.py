import sys
import traceback

import lib.puns as puns


def main(argv):
    punner = puns.Punner(config=puns.PunnerConfig())
    while True:
        try:
            topic = input("Topic > ")
            sentence = input("Sentence > ")

            print(punner.punnify(topic, sentence))
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
    print()


if __name__ == "__main__":
    main(sys.argv)
