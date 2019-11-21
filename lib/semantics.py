import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model
import json
import os
import os.path as path
import requests
import sys
import urllib.request as request
import zipfile


class SimilarWordMap:
    """
    Defines the base-level requirements for a SimilarWordMap. Must provide:
      1) Top k most similar words
      2) Similarity metric between two words
    """

    def get_similar_words(self, word, count):
        """
        Returns the top k (where  k = count) most similar words to the provided word.
        """
        raise NotImplementedError()


class FastTextSimilarWordMap(SimilarWordMap):
    DOWNLOAD_LINK = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    ZIP_FILE = "wiki.en.zip"
    MODEL_FILE = "wiki.en.bin"

    def __init__(self):
        """
        When constructed, this class will automatically donwload pretrained English word embeddings
        from Facebook's FastText and save them for later use.

        WARNING: The pretrained vector file is 6GB, so make sure you download it on reliable WiFi.
        """
        if not path.exists(self.ZIP_FILE) and not path.exists(self.MODEL_FILE):
            print("Downloading model...", file=sys.stderr)
            request.urlretrieve(self.DOWNLOAD_LINK, self.MODEL_FILE)

        if not path.exists(self.MODEL_FILE):
            print("Unzipping model...", file=sys.stderr)
            with zipfile.ZipFile(self.ZIP_FILE, "r") as zip:
                zip.extract(self.MODEL_FILE)

        print("Reading file...", file=sys.stderr)
        self.fb_model = load_facebook_model(path.join(os.getcwd(), self.MODEL_FILE))
        self.model = fb_model.wv

    def get_similar_words(self, word, count):
        return self.model.similar_by_word(word, topn=count)


class TwitterGloveSimilarWordMap(SimilarWordMap):
    def __init__(self):
        """
        Upon constructing this class, it loads up the Twitter GloVe word embeddings.
        """
        self.model = api.load("glove-twitter-25")

    def get_similar_words(self, word, count):
        return self.model.most_similar(word, topn=count)


class ServerSimilarWordMap(SimilarWordMap):
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 8080

    def __init__(self):
        assert b"OK!" == self.__request("healthcheck")

    def get_similar_words(self, word, count):
        data = json.loads(self.__request(path.join("similar_words", word, str(count))))

        return [(datum[0], datum[1]) for datum in data]

    def __request(self, url):
        return requests.get(
            path.join("http://{}:{}".format(self.SERVER_HOST, self.SERVER_PORT), url)
        ).content
