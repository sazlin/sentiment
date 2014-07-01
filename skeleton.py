import sys, os
import numpy as np
from operator import itemgetter as ig
from sklearn.linear_model import LogisticRegression as LR

vocab = [] #the features used in the classifier

#build vocabulary
def buildvocab():
    global vocab
    stopwords = open('stopwords.txt').read().lower().split()

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
    buildvocab()
    lr = make_classifier()
    test_classifier(lr)
