import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts  = ["reset my password", "order not arrived", "I want a refund", "track my shipment"]
labels = [0, 1, 2, 3]
responses = {
    0: "Go to settings and click Forgot Password.",
    1: "Your order arrives in 5-7 business days.",
    2: "Refunds are processed in 3-5 business days.",
    3: "Use your tracking ID on our website."
}

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1

seqs = tokenizer.texts_to_sequences(texts)
seqs = pad_sequences(seqs, maxlen=5, padding='post')
y    = tf.keras.utils.to_categorical(labels, 4)

model = models.Sequential([
    layers.Embedding(vocab_size, 8, input_length=5),
    layers.Bidirectional(layers.LSTM(16)),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(seqs, y, epochs=100, verbose=0)

def chat(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=5, padding='post')
    cat = np.argmax(model.predict(seq), axis=-1)[0]
    return responses[cat]

while True:
    inp = input("You: ")
    if inp == "exit":
        print("Bye!")
        break
    print("Chatbot:", chat(inp))