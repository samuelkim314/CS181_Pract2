import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import sklearn.linear_model as sklin
import sklearn.decomposition

def learn(X_train, y_train, mode='lsmr', reduction=None, n_components=10, alphas=[0.1, 1., 10.], normalize=False):
    def ridge():
        model = sklin.Ridge(normalize=normalize, alpha=alphas[0])
        model.fit(X_train, y_train)
        return (model.intercept_,model.coef_)

    def ridgeCV():
        model = sklin.RidgeCV(normalize=normalize, alphas=np.array(alphas))
        model.fit(X_train, y_train)
        print "FYI alpha = %f" % model.alpha_
        return (model.intercept_,model.coef_)

    def ARDRegression():
        model = sklin.ARDRegression()
        if sparse.issparse(X_train): X_t = X_train.toarray()
        else: X_t = X_train
        model.fit(X_t, y_train)
        return (model.intercept_,model.coef_)

    def BayesianRidge():
        model = sklin.BayesianRidge()
        if sparse.issparse(X_train): X_t = X_train.toarray()
        else: X_t = X_train
        model.fit(X_t, y_train)
        return (model.intercept_,model.coef_)

    def ElasticNetCV():
        model = sklin.ElasticNetCV()
        model.fit(X_train, y_train)
        return (model.intercept_,model.coef_)

    def Lasso():
        model = sklin.Lasso(alpha=alphas[0])
        model.fit(X_train, y_train)
        return (model.intercept_,model.coef_)

    def LassoCV():
        model = sklin.LassoCV(alpha=alphas[0])
        model.fit(X_train, y_train)
        return (model.intercept_,model.coef_)

    def LarsCV():
        model = sklin.LarsCV()
        if sparse.issparse(X_train): X_t = X_train.toarray()
        else: X_t = X_train
        model.fit(X_t, y_train)
        return (model.intercept_,model.coef_)

    def LassoLarsCV():
        model = sklin.LassoLarsCV()
        if sparse.issparse(X_train): X_t = X_train.toarray()
        else: X_t = X_train
        model.fit(X_t, y_train)
        return (model.intercept_,model.coef_)

    def LassoLarsIC():
        model = sklin.LassoLarsIC()
        if sparse.issparse(X_train): X_t = X_train.toarray()
        else: X_t = X_train
        model.fit(X_t, y_train)
        return (model.intercept_,model.coef_)

    def LinearRegression():
        model = sklin.LinearRegression()
        model.fit(X_train, y_train)
        return (model.intercept_,model.coef_)

    def LogisticRegression():
        model = sklin.LogisticRegression()
        model.fit(X_train, y_train)
        return (model.intercept_,model.coef_)

    def Perceptron():
        model = sklin.Perceptron()
        model.fit(X_train, y_train)
        print model.intercept_.shape
        print model.coef_.shape
        return (model.intercept_,model.coef_)

    def SGDRegressor():
        model = sklin.SGDRegressor()
        model.fit(X_train, y_train)
        return (model.intercept_,model.coef_)
        

    decomp = None
    if reduction=='tsvd':
        print "Decomposing with TruncatedSVD into %d components" % n_components
        decomp = sklearn.decomposition.TruncatedSVD(n_components=n_components)
        X_train = decomp.fit_transform(X_train)
    elif reduction == 'nmf':
        print "Decomposing with NMF into %d components" % n_components
        decomp = sklearn.decomposition.NMF(n_components=n_components, sparseness='data')
        X_train = decomp.fit_transform(X_train)


    modes = {
        'lsmr': lambda: (0,splinalg.lsmr(X_train,y_train)[0]),
        'lsqr': lambda: (0,splinalg.lsqr(X_train,y_train)[0]),
        'ridge': ridge,
        'ridgeCV': ridgeCV,
        'lasso':Lasso,
        'ARDRegression':ARDRegression,
        'BayesianRidge':BayesianRidge,
        'ElasticNetCV':ElasticNetCV,
        'LassoCV':LassoCV,
        'LarsCV':LarsCV,
        'LassoLarsCV':LassoLarsCV,
        'LassoLarsIC':LassoLarsIC,
        'LinearRegression':LinearRegression,
        'LogisticRegression':LogisticRegression,
        'Perceptron':Perceptron,
        'SGDRegressor':SGDRegressor

    }
    learned_w = modes[mode]()
    if decomp != None:
        return (learned_w, lambda X: decomp.transform(X))
    else:
        return learned_w
        