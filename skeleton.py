import os
from collections import OrderedDict
import numpy as np
from operator import itemgetter as ig
from sklearn.linear_model import LogisticRegression as LR

#vocab = [] #the features used in the classifier
vocab_list = []
stopwords_list = set([])
#build vocabulary
def build_vocab(n=100):
    vocab_dict = {}
    #global stopwords_list
    stopwords_list = set(open('stopwords.txt').read().lower().split())
    _populate_vocab_dict(
        vocab_dict,
        stopwords_list,
        os.listdir(os.getcwd() + "/pos"),
        "pos")
    _populate_vocab_dict(
        vocab_dict,
        stopwords_list,
        os.listdir(os.getcwd() + "/neg"),
        "neg")
    vocab_list = OrderedDict(sorted(vocab_dict.iteritems(), key=ig(1))[-n:]).keys()
    return vocab_list


def _populate_vocab_dict(d, stopwords, filenames, folder):
    for filename in filenames:
        print "Filename: ", filename
        with open(os.getcwd() + "/" + folder + "/" + filename) as f:
            words = f.read().lower().split()
            for word in words:
                if word not in stopwords:
                    if word in d:
                        d[word] += 1
                    else:
                        d[word] = 1
    ###TODO: Populate vocab list with N most frequent words in training data, minus stopwords


def vectorize(fn):
    global vocab
    vector = np.zeros(len(vocab))
    
    ###TODO: Create vector representation of 

    return vector

def make_classifier():
   
    #TODO: Build X matrix of vector representations of review files, and y vector of labels

    lr = LR()
    lr.fit(X,y)

    return lr

def test_classifier(lr):
    global vocab
    test = np.zeros((len(os.listdir('test')),len(vocab)))
    testfn = []
    i = 0
    y = []
    for fn in os.listdir('test'):
        testfn.append(fn)
        test[i] = vectorize(os.path.join('test',fn))
        ind = int(fn.split('_')[0][-1])
        y.append(1 if ind == 3 else -1)
        i += 1

    assert(sum(y)==0)
    p = lr.predict(test)

    r,w = 0,0
    for i,x in enumerate(p):
        if x == y[i]:
            r += 1
        else:
            w +=1
            print(testfn[i])
    print(r,w)


if __name__=='__main__':
    build_vocab()
    lr = make_classifier()
    test_classifier(lr)
