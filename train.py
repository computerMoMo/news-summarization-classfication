# -*- coding:utf-8 -*-

from __future__ import print_function
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Dropout, Activation, Input, LSTM, Bidirectional
from keras.models import Sequential
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate, average, maximum
from keras.constraints import max_norm
from keras.regularizers import l2
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping
from heapq import nlargest

import os
import numpy as np
import codecs

np.random.seed(1337)

global nb_words
global word_embedding_matrix
MAX_SEQUENCE_LENGTH = 25
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
DATA_DIR = 'data/'

train_texts_word = []
train_labels = []

test_texts_word = []
test_texts_char = []

labels_index = {'entertainment': 0,
                'sports': 1,
                'tech': 2,
                'world': 3,
                'finance': 4}

index_label = dict((v, k) for k, v in labels_index.iteritems())


# 定义神经网络模型
def single_cnn():

    model = Sequential()

    model.add(Embedding(nb_words+1,
                        EMBEDDING_DIM,
                        weights=[word_embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=True))
    model.add(Conv1D(filters=250,
                     kernel_size=3,
                     padding='valid',
                     activation='relu',
                     strides=1))

    model.add(MaxPooling1D(pool_size=model.output_shape[1]))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(len(labels_index), activation='softmax'))

    return model

if __name__ == '__main__':
    # 读入词向量，生成词向量的索引
    print ('indexing vectors')

    pre_trained_word_embeddings = Word2Vec.load(DATA_DIR+'news_summarization_300_dim.bin')
    word_weights = pre_trained_word_embeddings.wv.syn0

    word_embeddings_index = dict([(k, v.index) for k, v in pre_trained_word_embeddings.wv.vocab.items()])
    print('Found %s word vectors.' % len(word_embeddings_index))

    # 读入训练文本数据，生成训练数据中的词的索引
    print('processing train text data')

    with open(os.path.join(DATA_DIR, 'news-summarization-train.txt'), 'rb') as f:
        for line in f.readlines():
            train_texts_word.append(line.strip('\n').strip().split('\t')[1])
            train_labels.append(labels_index[line.strip('\n').strip().split('\t')[0]])

    print('Found %s words in train texts' % len(train_texts_word))

    tokenizer_word = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer_word.fit_on_texts(texts=train_texts_word)
    train_sequences_word = tokenizer_word.texts_to_sequences(texts=train_texts_word)

    train_word_index = tokenizer_word.word_index
    print('Found %s words token in train text' % len(train_word_index))

    # 生成神经网络用训练数据
    x_train = pad_sequences(sequences=train_sequences_word, maxlen=MAX_SEQUENCE_LENGTH)
    y_train = to_categorical(np.asarray(train_labels))
    print('Shape of train data tensor:', x_train.shape)
    print('Shape of train label tensor:', y_train.shape)

    # 生成词向量矩阵
    print('prepare embedding matrix')

    nb_words = min(MAX_NB_WORDS, len(train_word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

    for word, i in train_word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vectors = word_embeddings_index.get(word.decode('utf-8'))
        if embedding_vectors is not None:
            word_embedding_matrix[i] = word_weights[word_embeddings_index[word.decode('utf-8')], :]

    # 训练神经网络
    print('train model')

    check_pointer = ModelCheckpoint(filepath='models/best_model_weights.hdf5', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(patience=2)

    final_model = single_cnn()
    final_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    final_model_hist = final_model.fit(x=x_train, y=y_train, validation_split=0.1, epochs=100, batch_size=128,
                                       callbacks=[early_stopping, check_pointer])

    print('Results Summary')
    print('*' * 20)
    print('Best Accuracy:', np.max(final_model_hist.history['val_acc']))
    print('*' * 20)
