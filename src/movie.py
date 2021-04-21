import re, csv, string

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import nltk
from nltk.stem import *
try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords


stemmer = PorterStemmer()
STOPS = set(stopwords.words('english'))


def is_stopword(word):
    VALID = re.compile('^[a-zA-Z]{2,}$')
    return word in STOPS or len(word) <= 2 or not bool(VALID.match(word))


def data_file(data):
    assert data in ['train', 'valid', 'test'], f'invalid dataset: {data}'
    if data == 'train':
        return '../dataset/Train.csv'
    if data == 'valid':
        return '../dataset/Valid.csv'
    if data == 'test':
        return '../dataset/Test.csv'


def load_movie(data, data_limit, balanced_limit=False, load_class='both',
              train_ratio=0.7, stemmer=None):
    neg, pos = [], []
    with open(data_file(data), 'r', encoding="utf8") as file:
        reader = csv.reader(file)
        next(reader)
        for idx, line in enumerate(reader):
            line[0] = line[0].translate(
                str.maketrans('', '', string.punctuation))
            words = line[0].lower().split()
            words = set(words)
            if stemmer:
                words = [stemmer.stem(word) for word in line[0].split() if
                         not is_stopword(word)]
            else:
                words = [word for word in words if not is_stopword(word)]
            if line[1] == '1':
                if not balanced_limit:
                    pos.append(words)
                else:
                    if len(pos) < (data_limit // 2):
                        pos.append(words)
            else:
                if not balanced_limit:
                    neg.append(words)
                else:
                    if len(neg) < (data_limit // 2):
                        neg.append(words)

            if data_limit is not None:
                if (idx + 1) >= data_limit:
                    break

    if load_class == 'both':
        bows = pos + neg
        y = [1 for _ in range(len(pos))] + [0 for _ in range(len(neg))]
    if load_class == 'pos':
        bows = pos
        y = [1 for _ in range(len(pos))]
    if load_class == 'neg':
        bows = neg
        y = [0 for _ in range(len(neg))]

    words = np.concatenate(bows)
    word_counts = pd.Series(words).value_counts()
    selected_word = np.array(word_counts[word_counts > 10][5:].index)

    bows_onehot = []
    for bow in bows:
        bow = set(bow)
        one_hot = np.zeros_like(selected_word)
        for idx, word in enumerate(selected_word):
            if word in bow:
                one_hot[idx] = 1.
        bows_onehot.append(one_hot)

    x = np.r_[bows_onehot].astype(np.float)
    y = np.array(y)[..., np.newaxis].astype(np.int)

    return train_test_split(x, y, train_size=int(x.shape[0] * train_ratio))


if __name__ == '__main__':
    from model import HME

    x_train, x_test, y_train, y_test = load_movie('train', data_limit=5000,
                                                  train_ratio=0.8)
    n_feature = x_train.shape[1]
    n_level = 2

    n_expert1 = 1
    n_expert2 = 1
    max_iter = 3000
    stop_thre = np.inf

    hme = HME(n_feature, n_expert1, n_expert2, n_level, batch_size=20, lr=1.,
              l1_coef=0.0001, l21_coef=0.0001, algo='gd')

    hme.fit(x_train, y_train, max_iter=max_iter, stop_thre=stop_thre,
            log_interval=max_iter // 10)