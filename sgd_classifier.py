from IPython import parallel

import numpy
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score


def check_results(results):
    best = None
    best_a = None
    for alpha, res in results:
        if res > best:
            best = res
            best_a = alpha
    return best_a


def get_data():
    data = fetch_20newsgroups_vectorized(
        remove=('headers', 'footers', 'quotes')
    )
    return data


def get_MNB_with_alpha(alpha):
    #alpha, data = alphas
    clf = MultinomialNB(alpha)
    return alpha, numpy.mean(cross_val_score(clf, DATA.data, DATA.target, cv=10))


def calculate_results():
    data = get_data()
    alphas = [1E-4, 1E-3, 1E-2, 1E-1]
    #alphas = [(a, data) for a in alphas]
    clients = parallel.Client()
    clients.block = True
    dview = clients[:]
    dview['DATA'] = data
    lview = clients.load_balanced_view()
    with dview.sync_imports():
        import numpy
        from sklearn.datasets import fetch_20newsgroups_vectorized
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.cross_validation import cross_val_score
    results = []
    for idx, result in enumerate(lview.map(get_MNB_with_alpha, alphas)):
        results.append(result)
        print result
    return check_results(results)


if __name__ == '__main__':
    print calculate_results()
