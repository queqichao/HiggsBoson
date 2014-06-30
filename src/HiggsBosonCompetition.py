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

def read_test_data(file_name):
    ids = []
    features = []
    for line in open(file_name):
        l = split(",", line[:-1])
        ids.append(l[0])
        feature = []
        for f in l[1:]:
            feature.append(float(f))
        features.append(feature)
    return Data(ids, features, [], [])

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

def data_split(data, train_ratio, seed=None):
    n = len(data.features_)
    train_num = round(n*train_ratio)
    random.seed(seed)
    x = list(range(n))
    random.shuffle(x)

    train_ids = []
    train_features = []
    train_labels = []
    train_weights = []
    valid_ids = []
    valid_features = []
    valid_labels = []
    valid_weights = []

    for i in range(n):
        if i<train_num:
            train_ids.append(data.ids_[i])
            train_features.append(data.features_[i])
            train_labels.append(data.labels_[i])
            train_weights.append(data.weights_[i])
        else:
            valid_ids.append(data.ids_[i])
            valid_features.append(data.features_[i])
            valid_labels.append(data.labels_[i])
            valid_weights.append(data.weights_[i])

    return Data(train_ids, train_features, train_labels, train_weights),Data(valid_ids, valid_features, valid_labels, valid_weights)

def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
   
    assert s>=0
    assert b>=0 
    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return math.sqrt(radicand)

def optimize_AMS(sig_probs, labels, weights):
    s = 0
    b = 0
    threshold = 0
    for i in range(len(labels)):
        if labels[i] == -1:
            b += weights[i]
        else:
            s += weights[i]
    rank = [i[0] for i in sorted(enumerate(sig_probs), key=lambda x:x[1])]
 
    max_ams = AMS(s, b)
    for i in range(len(labels)):
        if i<len(labels)*0.9:
            if labels[rank[i]] == -1:
                b -= weights[rank[i]]
            else:
                s -= weights[rank[i]]
            if max_ams < AMS(s,b):
                threshold = sig_probs[rank[i]]
                max_ams = AMS(s,b)
    return threshold, max_ams

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

def train_adaboost(lr, n, data):
    ada_classifier = AdaBoostClassifier(learning_rate=lr, n_estimators=n, random_state=round(time.time()))
    ada_classifier.fit(np.array(data.features_), np.array(data.labels_))
    return ada_classifier

def eval_one_param_adaboost(lr, n, train, valid):
    ada_classifier = train_adaboost(lr, n, train)
    probs = ada_classifier.predict_proba(np.array(valid.features_))
    sig_probs = [probs[i][1] for i in range(len(probs))]
    threshold, max_ams = optimize_AMS(sig_probs, valid.labels_, valid.weights_)
   
    return max_ams

def num2label(l):
    if l==-1:
        return 'b'
    else:
        return 's'

def pred_adaboost(ada_classifier, test, threshold):
    test_probs = ada_classifier.predict_proba(np.array(test.features_)).tolist()
    test_sig_probs = [test_probs[i][1] for i in range(len(test_probs))]
    preds = []
    for x in test_sig_probs:
        if x > threshold:
            preds.append(1)
        else:
            preds.append(-1)
    return preds,test_sig_probs

def compute_result(lr, n, train_data, test_data, filename):
    train,valid = data_split(train_data, 0.9, time.time())
    ada_classifier = train_adaboost(lr, n, train)
    valid_probs = ada_classifier.predict_proba(np.array(valid.features_)).tolist()
    sig_probs = [valid_probs[i][1] for i in range(len(valid_probs))]
    threshold, max_ams = optimize_AMS(sig_probs, valid.labels_, valid.weights_)
    preds,test_sig_probs = pred_adaboost(ada_classifier, test_data, threshold)
    log_prob = [math.log(x) for x in test_sig_probs]
    rank = compute_rank([log_prob[j] for j in range(len(log_prob))])
    rank = [r+1 for r in rank]
    f = open(filename, 'w')
    for i in range(len(preds)):
        f.write('%s,%d,%s\n' % (test_data.ids_[i], rank[i], num2label(preds[i])))
    f.close()

def compute_rank(scores):
    l = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1])]
    return [i[0] for i in sorted(enumerate(l), key=lambda x:x[1])] 
