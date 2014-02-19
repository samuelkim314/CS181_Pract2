import numpy as np

def basisPoly(x, M):
    """phi_j(x) = x^j, where j goes from 0 to M-1
    if x is a single number, returns np.array of phi_j(x)
    if x is an array of length N, returns the NxM design matrix
    """
    ms = np.arange(M)

    if type(x) is float:
        return np.power(x, ms)
    else:
        return np.power(x[:, np.newaxis], ms[np.newaxis, :])

def basisFourier(x, M):
    Mcos = math.trunc((M - 1) / 2)
    Msin = M - 1 - Mcos
    msCos = np.arange(Mcos)
    msSin = np.arange(Msin)

    period = np.amax(x) - np.amin(x)

    if type(x) is float:
        sinSub = numpy.sin(x * msSin * 2 * np.pi / period)
        cosSub = numpy.cos(x * msSin * 2 * np.pi / period)
        return np.concatenate((np.array([1]), sinSub, cosSub), axis=0)
    else:
        return np.power(x[:, np.newaxis], ms[np.newaxis, :])

def errorFun(x, t, w, basis):
    return 0.5*np.sum((t - np.dot(w, basis(x)))**2)

def solveW(x, t, basis, M):
    """Solves exactly for w, the weights. Not recommended for large data sets.
    Arguments:
        x:     array of inputs
        t:     array of corresponding target variables
        basis: function that calculates basis functions for inputs
        M:     number of basis functions to use
    Return:
        w:    array of parameters for the basis functions"""
    #design matrix given by phi_nj = phi_j(x_n)
    phiMat = basis(x, M)

    #calculates the Moore-Penrose pseudo-inverse of the matrix phi
    phiTphi = np.dot(np.transpose(phiMat), phiMat)
    MPMat = np.dot(np.linalg.inv(phiTphi), np.transpose(phiMat))
    return np.dot(MPMat, t)
