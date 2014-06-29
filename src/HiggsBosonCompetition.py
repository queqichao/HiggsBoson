# data processing module

from re import *
import random
import math
import svmutil
import numpy as np
import time
from sklearn.ensemble import AdaBoostClassifier

class Data:
    ids_ = []
    features_ = []
    labels_ = []
    weights_ = []

    def __init__(self, ids, features, labels, weights):
        self.ids_ = ids
        self.features_ = features
        self.labels_ = labels
        self.weights_ = weights

def read_data(file_name):
    ids = []
    features = []
    labels = []
    weights = []
    for line in open(file_name):
        l = split(",", line[:-1])
        ids.append(l[0])
        feature = []
        for f in l[1:-2]:
            feature.append(float(f))
        features.append(feature)
        weights.append(float(l[-2]))
        if l[-1]=='s':
            labels.append(1)
        else:
            labels.append(-1)
    return Data(ids, features, labels, weights)

def data_partition(n, k, seed=None):
    random.seed(seed)
    x = list(range(n))
    partitions = []
    sub_size = math.floor(n/k)
    random.shuffle(x)
    count = 0;
    for i in range(k):
        subset = []
        for j in range(sub_size):
            subset.append(x[count])
            count += 1
        if i < n%k:
            subset.append(x[count])
            count += 1
        partitions.append(subset)
        
    return partitions

def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
    
    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return math.sqrt(radicand)

def eval_AMS(preds, labels, weights):
    s = 0;
    b = 0;
    if len(preds) != len(labels):
        print("Predicts and labels should be of the same length.")
    for i in range(len(preds)):
        if preds[i] == 1:
            if labels[i] == -1:
                b += weights[i]
            else:
                s += weights[i]
    return AMS(s, b)

def eval_one_param(gamma, c, data, partitions):
    preds = data.labels_;
    for i in range(len(partitions)):
        training_data = []
        training_labels = []
        testing_data = []
        testing_labels = []
        for j in range(len(partitions)):
            for k in range(len(partitions[j])):
                if j==i:
                    testing_data.append(data.features_[partitions[j][k]])
                    testing_labels.append(data.labels_[partitions[j][k]])
                else:
                    training_data.append(data.features_[partitions[j][k]])
                    training_labels.append(data.labels_[partitions[j][k]])
        m = svmutil.svm_train(training_labels, training_data, '-t 2 -c '+'%.4f' % c +' -g '+'%.4f' % gamma)
        (pred, p_acc, p_vals) = svmutil.svm_pred(testing_labels, testing_data, m)
        for k in range(len(partitions[i])):
            preds[partitions[i][k]] = pred[k]
    return eval_AMS(preds, data.labels_, data.weights_)

def eval_one_param_adaboost(lr, n, data):
    ada_classifier = AdaBoostClassifier(learning_rate=lr, n_estimators=n, random_state=round(time.time()))
    ada_classifier.fit(np.array(data.features_), np.array(data.labels_))
    preds = ada_classifier.predict(np.array(data.features_))
    return eval_AMS(preds.tolist(), data.labels_, data.weights_)
    
