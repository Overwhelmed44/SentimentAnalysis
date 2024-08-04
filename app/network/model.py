from keras.api.layers import Embedding, Dense, GRU
from . import MAX_WORDS_COUNT, MAX_INPUT_LENGTH
from keras.api import Sequential

model = Sequential([
    Embedding(MAX_WORDS_COUNT, 256, input_length=MAX_INPUT_LENGTH),
    GRU(256, return_sequences=True),
    GRU(128),
    Dense(2, activation='softmax')
])

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
