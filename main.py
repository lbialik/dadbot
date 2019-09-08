import sys

from typing import List


def main(argv: List[str]) -> None:
    print("Hello {}".format(argv[1]))


if __name__ == "__main__":
    main(sys.argv)
