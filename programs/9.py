import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

text = [
    "the sun rises in the east",
    "the sun sets in the west",
    "machine learning is powerful",
    "deep learning is interesting",
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

total_words = len(tokenizer.word_index) + 1

seq = []
for sentence in text:
    tokens = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokens)):
        seq.append(tokens[:i+1])

max_len = max(len(x) for x in seq)
seq = pad_sequences(seq, maxlen=max_len, padding='pre')

X = seq[:, :-1]
y = np.eye(total_words)[seq[:, -1]]

model = Sequential([
    Embedding(total_words, 10, input_length=max_len-1),
    SimpleRNN(50),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, epochs=100, verbose=1)

def next_word(text_input):
    token = tokenizer.texts_to_sequences([text_input])[0]
    token = pad_sequences([token], maxlen=max_len-1, padding='pre')

    pred = model.predict(token, verbose=0)
    return tokenizer.index_word[np.argmax(pred)]

print("Next word:", next_word("machine learning "))