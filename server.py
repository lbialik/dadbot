from flask import Flask
import json

import lib.semantics as semantics

app = Flask(__name__)
model = semantics.TwitterGloveSimilarWordMap()


@app.route("/healthcheck")
def healthcheck():
    return "OK!"


@app.route("/similar_words/<word>/<int:count>")
def similar_words(word, count):
    return json.dumps(model.get_similar_words(word, count))


if __name__ == "__main__":
    app.run(
        host=semantics.ServerSimilarWordMap.SERVER_HOST,
        port=semantics.ServerSimilarWordMap.SERVER_PORT,
    )
