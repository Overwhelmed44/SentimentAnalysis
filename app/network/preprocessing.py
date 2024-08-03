from keras.api.utils import pad_sequences
from . import tokenizer, MAX_INPUT_LENGTH
import re


def filter_text(text):
    return re.sub(r'[^а-яА-Я ]*', '', text)


def texts_to_sequences(texts):
    return pad_sequences(tokenizer.texts_to_sequences(texts), MAX_INPUT_LENGTH)
