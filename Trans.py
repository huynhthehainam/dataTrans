import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import io
import string
import re
tf.enable_eager_execution()

def preprocess(sentence):
    sent = sentence.lower()
    sent = sent.strip()
    sent = "<start> " + sent + " <end>"
    return sent

X_train = []
X_test = []
with open('./data/train.en','r',encoding="utf8") as f:
    rawData = f.read().strip().split("\n")
for line in rawData:
    line = preprocess(line)
    X_train.append(line)
with open('./data/tst2013.en','r',encoding="utf8") as f:
    rawData = f.read().strip().split("\n")
for line in rawData:
    line = preprocess(line)
    X_test.append(line)

Y_train = []
Y_test = []
with open('./data/train.vi','r',encoding="utf8") as f:
    rawData = f.read().strip().split("\n")
for line in rawData:
    line = preprocess(line)
    Y_train.append(line)
with open('./data/tst2013.vi','r',encoding="utf8") as f:
    rawData = f.read().strip().split("\n")
for line in rawData:
    line = preprocess(line)
    Y_test.append(line)

class Language():
    def __init__(self, lines):
        self.lines = lines
        self.word2id = {}
        self.id2word = {}
        self.vocab = set()
        self.max_len = 0
        self.min_len = 0
        self.vocab_size = 0
        self.init_language_params()

    def init_language_params(self):
        for line in self.lines:
            self.vocab.update(line.split(" "))
        self.word2id['<pad>'] = 0
        for id, word in enumerate(self.vocab):
            self.word2id[word] = id + 1
        for word, id in self.word2id.items():
            self.id2word[id] = word
        self.max_len = max([len(line.split(" ")) for line in self.lines])
        self.min_len = min([len(line.split(" ")) for line in self.lines])
        self.vocab_size = len(self.vocab) + 1
            
    def sentence_to_vector(self, sent):
        return np.array([self.word2id[word] for word in sent.split(" ")])
            
    def vector_to_sentence(self, vector):
        return " ".join([self.id2word[id] for id in vector])

inp_lang = Language(X_train)
tar_lang = Language(Y_train)
inp_test_lang=Language(X_test)
tar_test_lang=Language(X_test)

inp_vector = [inp_lang.sentence_to_vector(line) for line in inp_lang.lines]
tar_vector = [tar_lang.sentence_to_vector(line) for line in tar_lang.lines]
inp_test_vector = [inp_test_lang.sentence_to_vector(line) for line in inp_test_lang.lines]
tar_test_vector = [tar_test_lang.sentence_to_vector(line) for line in tar_test_lang.lines]

inp_tensor = tf.keras.preprocessing.sequence.pad_sequences(inp_vector, inp_lang.max_len, padding='post')
tar_tensor = tf.keras.preprocessing.sequence.pad_sequences(tar_vector, tar_lang.max_len, padding='post')
inp_test_tentor = tf.keras.preprocessing.sequence.pad_sequences(inp_test_vector, inp_test_lang.max_len, padding='post')
tar_test_tentor = tf.keras.preprocessing.sequence.pad_sequences(tar_test_vector, tar_test_lang.max_len, padding='post')
print(inp_tensor.shape, tar_tensor.shape)

BATCH_SIZE = 32
BUFFER_SIZE = inp_tensor.shape[0]
N_BATCH = BUFFER_SIZE//BATCH_SIZE
hidden_unit = 1024
embedding_size = 256
print(BUFFER_SIZE)


dataset = tf.data.Dataset.from_tensor_slices((inp_tensor, tar_tensor))
dataset = dataset.batch(BATCH_SIZE)

tmp_x, tmp_y = next(iter(dataset))
print(tmp_x.shape)
print(tmp_y.shape)

class Encode(tf.keras.Model):
    def __init__(self, embedding_size, vocab_size, hidden_units):
        super(Encode, self).__init__()
        self.Embedding = tf.keras.layers.Embedding(vocab_size,embedding_size)
        self.GRU = tf.keras.layers.GRU(
            hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')
        self.hidden_units = hidden_units
        
    def call(self, x, hidden_state):
        try:
            x = self.Embedding(x)
        except:
            print(x, print(inp_lang.vocab_size))          
        outputs, last_state = self.GRU(x, hidden_state)
        return outputs, last_state
    
    def init_hidden_state(self, batch_size):
        return tf.zeros([batch_size, self.hidden_units])

encoder = Encode(embedding_size, inp_lang.vocab_size, hidden_unit)
hidden_state = encoder.init_hidden_state(BATCH_SIZE)
tmp_outputs, last_state = encoder(tmp_x, hidden_state)
print(tmp_outputs.shape)
print(last_state.shape)

class Attention(tf.keras.Model):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        self.W_out_encode = tf.keras.layers.Dense(hidden_unit)
        self.W_state = tf.keras.layers.Dense(hidden_unit)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, encode_outs, pre_state):
        pre_state = tf.expand_dims(pre_state, axis=1)
        pre_state = self.W_state(pre_state)
        encode_outs = self.W_out_encode(encode_outs)
        score = self.V(
            tf.nn.tanh(
                pre_state + encode_outs)
        )
        score = tf.nn.softmax(score, axis=1)
        context_vector = score*encode_outs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, score

attention = Attention(hidden_unit)
context_vector, attention_weight = attention(tmp_outputs, last_state)
print(context_vector.shape, attention_weight.shape)

class Decode(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_units):
        super(Decode, self).__init__()
        self.hidden_units = hidden_units
        self.Embedding = tf.keras.layers.Embedding(vocab_size,embedding_size)
        self.Attention = Attention(hidden_units)
        self.GRU = tf.keras.layers.GRU(
            hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.Fc = tf.keras.layers.Dense(vocab_size)
            
    def call(self, x, encode_outs, pre_state):
        x = tf.expand_dims(x, axis=1)
        try:
            x = self.Embedding(x)
        except:
            print(x, print(tar_lang.vocab_size))          
        context_vector, attention_weight = self.Attention(encode_outs, pre_state)
        context_vector = tf.expand_dims(context_vector, axis=1)
        gru_inp = tf.concat([x, context_vector], axis=-1)
        out_gru, state = self.GRU(gru_inp)
        out_gru = tf.reshape(out_gru, (-1, out_gru.shape[2]))
        return self.Fc(out_gru), state
    
    
decode = Decode(tar_lang.vocab_size, embedding_size, hidden_unit)
print(last_state.shape, tmp_outputs.shape, tmp_y[:, 0].shape)
decode_out, state = decode(tmp_y[:, 0], tmp_outputs, last_state)

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

EPOCHS = 100
optimizer = tf.train.AdamOptimizer()
encoder = Encode(embedding_size, vocab_size=inp_lang.vocab_size, hidden_units=hidden_unit)
decoder = Decode(vocab_size=tar_lang.vocab_size, embedding_size=embedding_size, hidden_units=hidden_unit)

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_id, (x, y) in enumerate(dataset.take(N_BATCH)):
        loss = 0
        with tf.GradientTape() as tape:
            
            first_state = encoder.init_hidden_state(batch_size=BATCH_SIZE)
            encode_outs, last_state = encoder(x, first_state)
            decode_state = last_state
            decode_input = [tar_lang.word2id["<start>"]]*BATCH_SIZE
            
            for i in range(1, y.shape[1]):
                decode_out, decode_state = decoder(decode_input, encode_outs, decode_state)
                loss += loss_function(y[:, i], decode_out)
                decode_input = y[:, i]
                
            train_vars = encoder.trainable_variables + decoder.trainable_variables
            grads = tape.gradient(loss, train_vars)
            optimizer.apply_gradients(zip(grads, train_vars))
        total_loss += loss
        print("helo")
    print(total_loss.numpy())

