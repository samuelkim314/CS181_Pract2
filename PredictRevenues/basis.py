import numpy as np
from scipy import sparse

def poly(Xmat, M=2):
    """input: a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds"""
    ones = sparse.csr_matrix(np.ones((Xmat.shape[0],1)))

    polyMat = sparse.hstack([ones, Xmat])
    if M > 2:
        for i in (range(2,M)):
            Xtemp = Xmat.copy()
            Xtemp.data **= i
            polyMat = sparse.hstack([polyMat, Xtemp])
    return polyMat

def polydot(Xtest, w):
    pass