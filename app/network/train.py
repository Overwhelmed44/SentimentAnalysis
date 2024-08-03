from .preprocessing import texts_to_sequences, filter_text
from app.utils import tskv_reader, json_write
from keras.api.utils import to_categorical
from . import tokenizer, model
from itertools import islice


def train(n=10_000):
    iter_ = tskv_reader(r'app/network/dataset/dataset.tskv')

    revs = [[''] * n, [0] * n]
    for it, rev in enumerate(islice(iter_, n)):
        revs[0][it] = filter_text(rev['text'])

        rating = int(rev['rating'][0])
        revs[1][it] = 0 if rating <= 3 else 1

    tokenizer.fit_on_texts(revs[0])

    X = texts_to_sequences(revs[0])
    Y = to_categorical(revs[1], 2)

    model.fit(X, Y, 32, 10)

    model.save(r'app/network/example/model.keras')
    json_write(tokenizer.to_json(), r'app/network/example/tokenizer.json')
