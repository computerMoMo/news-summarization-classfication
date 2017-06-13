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
Test_data_path = '/data/cn_corpus/news-summarization/train.article.txt'
# Test_data_path = 'data/news-summarization-train.txt'
model_path = 'models/best_model_weights.hdf5'

train_texts_word = []

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
    # 读入词向量
    print('indexing vectors')
    pre_trained_word_embeddings = Word2Vec.load(DATA_DIR + 'news_summarization_300_dim.bin')
    word_weights = pre_trained_word_embeddings.wv.syn0

    word_embeddings_index = dict([(k, v.index) for k, v in pre_trained_word_embeddings.wv.vocab.items()])
    print('Found %s word vectors.' % len(word_embeddings_index))

    # 打开训练文本文件,生成word index 词典
    print('processing train text data')

    with open(os.path.join(DATA_DIR, 'news-summarization-train.txt'), 'rb') as f:
        for line in f.readlines():
            train_texts_word.append(line.strip('\n').strip().split('\t')[1])

    print('Found %s words in train texts' % len(train_texts_word))

    tokenizer_word = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer_word.fit_on_texts(texts=train_texts_word)
    train_word_index = tokenizer_word.word_index

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

    # 打开带预测文件
    print('predicting test data')
    file_test_reader = open(Test_data_path, 'rb')
    file_result = open(DATA_DIR+'result.txt', 'wb')
    file_log = open(DATA_DIR+'log.txt', 'wb')

    cnn_model = single_cnn()
    cnn_model.load_weights(model_path)

    total_nums = 0
    end_flag = False
    while True:
        epoch_nums = 0
        test_text_word = []
        while epoch_nums < 10000:
            str_line = file_test_reader.readline()
            if not str_line:
                end_flag = True
                break
            test_text_word.append(str_line.strip('\n').strip())
            epoch_nums += 1
            total_nums += 1

        test_text_sequences = tokenizer_word.texts_to_sequences(texts=test_text_word)
        x_test_data = pad_sequences(sequences=test_text_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        y_test_result = cnn_model.predict(x=x_test_data)

        result_lines = []
        for idx, item in enumerate(y_test_result):
            top_3 = nlargest(3, range(len(labels_index)), key=lambda j: item[j])
            line = index_label[top_3[0]] + '\t' + index_label[top_3[1]] + '\t' + index_label[top_3[2]] + '\n'
            result_lines.append(line)

        file_result.writelines(result_lines)
        file_log.write(str(total_nums)+'\n')
        if end_flag:
            break

    file_test_reader.close()
    file_result.close()
    file_log.close()
    print('program finished!')
