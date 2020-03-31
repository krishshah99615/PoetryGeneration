import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.models import load_model
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
model = load_model('model.h5')



def pre(s):
    
    global model
    t = s
    n = 100
    for i in range(n):
        
        tl = tokenizer.texts_to_sequences([t])[0]
        p = pad_sequences([tl], maxlen=max_len-1, padding='pre')
        predic = model.predict_classes(p)
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predic:
                output_word = word
                break
        t += " "+output_word
        if i % 6 == 0:
            t = t+"\n"
    return t.split('\n')
