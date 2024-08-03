from keras._tf_keras.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.api.models import load_model
from app.utils import json_read
from os.path import exists

MAX_WORDS_COUNT = 5000
MAX_INPUT_LENGTH = 12

if exists(t := 'app/network/example/tokenizer.json'):
    tokenizer = tokenizer_from_json(json_read(t))
else:
    tokenizer = Tokenizer(MAX_WORDS_COUNT)

if exists(m := 'app/network/example/model.keras'):
    model = load_model(m)
else:
    from .model import model
    from .train import train

    train()
