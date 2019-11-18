import sys
import lib.semantics as semantics


def main(argv):
    word_similarity_map = semantics.TwitterGloveSimilarWordMap()
    while True:
        word = input("> ")
        for similar in word_similarity_map.get_similar_words(word, 10):
            print("    {}".format(similar))


if __name__ == "__main__":
    main(sys.argv)
