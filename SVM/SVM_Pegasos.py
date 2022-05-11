import os
import numpy as np
import random
import csv
from collections import Counter
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

def folder_list(path,label):
    """
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    """
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    """
    Read each file into a list of strings.
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on',
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    """
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(str.maketrans("", "", symbols)).strip(), lines)
    words = filter(None, words)
    return list(words)

def load_and_shuffle_data():
    """
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    """
    pos_path = "data_reviews/pos"
    neg_path = "data_reviews/neg"

    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)

    review = pos_review + neg_review
    random.shuffle(review)      
    return review

# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code
def dotProduct(d1, d2):
    """
    Return dot product of values of two dicts.
    Args:
        dict d1 - a feature vector represented by a mapping from a feature (string) to a weight (float).
        dict d2 - same as d1
    Returns:
        float - the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def bag_of_words(words):
    """
    Converts an example (a list of words) into a sparse bag-of-words representation.
    Args:
        words - an example/list of words
    Returns:
        sparse_bag_of_words - dict of sparse representation of words
    """
    sparse_bag_of_words = dict(Counter(words))
    return sparse_bag_of_words

def load_and_split_data(test_size=0.25):
    """
    Load, shuffle and split data.
    Args:
        test_size - the ratio of testing examples
    Returns:
        X_train - a list of dictionaries of training examples
        y_train - a list of corresponding labels of training examples
        X_test - a list of dictionaries of testing examples
        y_test - a list of corresponding labels of testing examples
    """
    data = load_and_shuffle_data()
    X = [data[i][:-2] for i in range(2000)]
    y = [data[i][-1] for i in range(2000)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train = list(map(bag_of_words, X_train))#convert list of words into sparse bag_of_words representation
    X_test = list(map(bag_of_words, X_test))
    return X_train, y_train, X_test, y_test

class SVMPegasos1:
    """
    Implement of SVM with SGD Pegasos Algorithm.
    """
    def __init__(self, T, lambda1, w_0):
        self.T = T
        self.lambda1 = lambda1
        self.w_0 = w_0

    def fit(self, X, y, num_epoch):
        m = len(X)
        w = self.w_0
        t = 0
        for epoch in range(num_epoch):
            for j in range(self.T):
                t += 1
                eta = 1 / (self.lambda1 * (t+1))
                i = np.random.choice(np.arange(0, m))
                X_i = X[i]
                y_i = y[i]
                score = dotProduct(w, X_i)
                if y_i * score < 1:
                    increment(w, (-eta ) * self.lambda1, w)
                    increment(w, eta * y_i, X_i)
                else:
                    increment(w, (-eta ) * self.lambda1, w)
        return w

class SVMPegasos2:
    """
    Implement of SVM with SGD with Pegasos Algorithm.
    """
    def __init__(self, T, lambda1, w_0):
        self.T = T
        self.lambda1 = lambda1
        self.w_0 = w_0

    def fit(self, X, y, num_epoch):
        m = len(X)
        w = self.w_0
        s = 1.0
        t = 0
        for epoch in range(num_epoch):
            for j in range(self.T):
                t += 1
                eta = 1 / (self.lambda1 * (t+1))
                i = np.random.choice(np.arange(0, m))
                X_i = X[i]
                y_i = y[i]
                score = dotProduct(w, X_i)
                if y_i * score < 1:
                    s = (1 - eta * self.lambda1) * s
                    increment(w, (eta * y_i)/s, X_i)
                else:
                    increment(w, (-eta ) * self.lambda1, w)
        return w





