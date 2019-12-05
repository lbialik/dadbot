from flask import Flask
from flask import request
from flask import Response
import json

from lib import puns
from lib import semantics

app = Flask(__name__)
model = semantics.TwitterGloveSimilarWordMap()


@app.route("/healthcheck")
def healthcheck():
    """
    Used when a consumer of this service launches to check whether or not the
    embedding service is live.
    """

    return "OK!"


@app.route("/similar_words/<word>/<int:count>")
def similar_words(word, count):
    """
    Returns a list of candidate words similar to the provided word with length
    equal to the provided count.
    """

    resp = Response(json.dumps(model.get_similar_words(word, count)))
    resp.headers.add("Content-Type", "application/json")

    return resp


@app.route("/generate_pun", methods=["POST"])
def generate_pun():
    """
    Generates a pun given a topic and a sentence.
    """
    punner = puns.Punner()

    resp = Response(
        json.dumps(
            {
                "sentence": punner.punnify(
                    request.json["topic"], request.json["sentence"]
                )
            }
        )
    )
    resp.headers.add("Content-Type", "application/json")

    return resp


if __name__ == "__main__":
    app.run(
        host=semantics.ServerSimilarWordMap.SERVER_HOST,
        port=semantics.ServerSimilarWordMap.SERVER_PORT,
    )
