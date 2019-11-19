import sys
import lib.puns as puns


def main(argv):
    punner = puns.Punner(config=puns.PunnerConfig())
    try:
        while True:
            topic = input("Topic > ")
            sentence = input("Sentence > ")

            print(punner.punnify(topic, sentence))
    except (EOFError, KeyboardInterrupt):
        print()
        return


if __name__ == "__main__":
    main(sys.argv)
