import os
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import CountVectorizer as CV


def build_vocab(n=100):
    stopwords = open('stopwords.txt').read().lower().split()
    pos_file_list = os.listdir(os.getcwd() + "/pos")
    neg_file_list = os.listdir(os.getcwd() + "/neg")
    labels_list = []
    [labels_list.append(1) for file in pos_file_list]
    [labels_list.append(-1) for file in neg_file_list]
    file_list = \
        [os.getcwd() + '/pos/' + filename for filename in pos_file_list] +\
        [os.getcwd() + '/neg/' + filename for filename in neg_file_list]

    vec = CV(input='filename',
        analyzer='word',
        stop_words=stopwords,
        max_features=n)
    print "Building X, Y..."
    X = vec.fit_transform(file_list).toarray()
    Y = np.array(labels_list)
    print "Done"
    return X, Y, vec.get_feature_names()


def run_classifier(lr, vocab):
    test_vec = CV(input='filename',
                         analyzer='word',
                         vocabulary=vocab)
    test_file_list = \
        [os.getcwd() + '/test/' + filename for filename in os.listdir(os.getcwd() + "/test")]
    print "Fitting transform..."
    test_matrix = test_vec.fit_transform(test_file_list, vocab).toarray()
    print "Done"
    i = 0
    test_file_known_sentiment = []
    for filename in test_file_list:
        indicator = int(filename.split('_')[0][-1])
        test_file_known_sentiment.append(1 if indicator == 3 else -1)
        i += 1

    assert sum(test_file_known_sentiment) == 0
    predications = lr.predict(test_matrix)

    correct, wrong = 0, 0
    for i, predicated_sentiment in enumerate(predications):
        if predicated_sentiment == test_file_known_sentiment[i]:
            correct += 1
        else:
            wrong += 1
            print test_file_list[i]
    print correct, wrong
    return correct / (float(correct) + wrong)


if __name__ == '__main__':
    x, y, vocab = build_vocab(10000)
    print "Vocab: ", vocab
    lr = LR()
    lr.fit(x, y)  # Setup classifier wth our matrix (x) and labels (y)
    run_classifier(lr, vocab)
