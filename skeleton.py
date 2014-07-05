import os
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import BernoulliNB as BNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import LinearSVC as SVM
from sklearn.feature_extraction.text import CountVectorizer as CV

def assemble_inputs():
    stopwords = open('stopwords.txt').read().lower().split()
    pos_file_list = os.listdir(os.getcwd() + "/pos")
    neg_file_list = os.listdir(os.getcwd() + "/neg")
    labels_list = []
    [labels_list.append(1) for file in pos_file_list]
    [labels_list.append(-1) for file in neg_file_list]
    file_list = \
        [os.getcwd() + '/pos/' + filename for filename in pos_file_list] +\
        [os.getcwd() + '/neg/' + filename for filename in neg_file_list]
    return stopwords, file_list, labels_list


def build_word_matrix(n=100):
    stopwords, file_list, labels_list = assemble_inputs()
    vec = CV(input='filename',
        analyzer='word',
        stop_words=stopwords,
        max_features=n)
    print "Building X, Y..."
    matrix = vec.fit_transform(file_list).toarray()
    Y = np.array(labels_list)
    print "Done"
    return matrix, Y, vec.get_feature_names()


def build_test_matrix(vocab):
    test_vec = CV(input='filename',
        analyzer='word',
        vocabulary=vocab)
    test_file_list = \
        [os.getcwd() + '/test/' + filename for filename in os.listdir(os.getcwd() + "/test")]
    test_matrix = test_vec.fit_transform(test_file_list, vocab).toarray()
    return test_matrix, test_file_list


def run_classifier(classifier, vocab):
    test_matrix, test_file_list = build_test_matrix(vocab)
    i = 0
    test_file_known_sentiment = []
    for filename in test_file_list:
        indicator = int(filename.split('_')[0][-1])
        test_file_known_sentiment.append(1 if indicator == 3 else -1)
        i += 1

    assert sum(test_file_known_sentiment) == 0
    predictions = classifier.predict(test_matrix)

    correct, wrong = 0, 0
    for i, predicated_sentiment in enumerate(predictions):
        if predicated_sentiment == test_file_known_sentiment[i]:
            correct += 1
        else:
            wrong += 1
    #print correct, wrong
    return correct / (float(correct) + wrong)


if __name__ == '__main__':
    matrix, y, vocab = build_word_matrix(1000)
    ### compares three different methods: Logistic Regression, Naive Bayes,
    ### and Support Vector Machine
    lr = LR()
    lr.fit(matrix, y)  # Setup classifier wth our matrix (x) and labels (y)
    print "LR: ", run_classifier(lr, vocab)
    nb = BNB()
    nb.fit(matrix, y)
    print "NB: ", run_classifier(nb, vocab)
    svm = SVM()
    svm.fit(matrix, y)
    print "SVM: ", run_classifier(svm, vocab)
    mnb = MNB()
    mnb.fit(matrix, y)
    print "MNB: ", run_classifier(mnb, vocab)

