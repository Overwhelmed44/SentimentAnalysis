from .network.preprocessing import texts_to_sequences, filter_text
from . import model


def evaluate_text(text, as_text=True):
    return ['Отрицательный отзыв', 'Положительный отзыв'][p := model(texts_to_sequences([filter_text(text)]))[0].numpy().argmax()] if as_text else p
