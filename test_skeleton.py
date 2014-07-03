from sklearn.linear_model import LogisticRegression as LR

from skeleton import build_vocab, run_classifier


def test_movie_sent_analyzer_n_100():
    x, y, vocab = build_vocab(100)
    lr = LR()
    lr.fit(x, y)
    percent_right = run_classifier(lr, vocab)
    assert percent_right > 0.70


def test_movie_sent_analyzer_n_1000():
    x, y, vocab = build_vocab(1000)
    lr = LR()
    lr.fit(x, y)
    percent_right = run_classifier(lr, vocab)
    assert percent_right > 0.82


def test_movie_sent_analyzer_n_10000():
    x, y, vocab = build_vocab(10000)
    lr = LR()
    lr.fit(x, y)
    percent_right = run_classifier(lr, vocab)
    assert percent_right > 0.85
