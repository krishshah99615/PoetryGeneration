import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy


data = open('data.txt').read()

tokenizer = Tokenizer(oov_token="<OOV>")
sent = data.lower().split('\n')
tokenizer.fit_on_texts(sent)
total_words = len(tokenizer.word_index)+1
word_index = tokenizer.word_index
input_seq = []
for line in sent:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        ng = token_list[:i+1]
        input_seq.append(ng)
max_len = max([len(x) for x in input_seq])
input_seq = np.array(pad_sequences(input_seq, maxlen=max_len, padding='pre'))
xs, labels = input_seq[:, :-1], input_seq[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = keras.models.Sequential(
    [
        keras.layers.Embedding(total_words, 100, input_length=max_len-1),
        keras.layers.Bidirectional(keras.layers.LSTM(150)),
        keras.layers.Dense(total_words, activation='softmax')
    ]
)
adam = keras.optimizers.Adam(lr=0.01)
model.compile(loss=categorical_crossentropy,
              optimizer='adam', metrics=['accuracy'])
model.fit(xs, ys, epochs=100, verbose=1)
model.save('model.h5')
