import numpy as np

def basisPoly(x, M):
    """phi_j(x) = x^j, where j goes from 0 to M-1
    if x is a single number, returns np.array of phi_j(x)
    if x is an array of length N, returns the NxM design matrix
    """
    ms = np.arange(M)

    if type(x) is int:
        return np.power(x, ms)
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
    phiMat = basisPoly(x, M)

    #calculates the Moore-Penrose pseudo-inverse of the matrix phi
    phiTphi = np.dot(np.transpose(phiMat), phiMat)
    MPMat = np.dot(np.linalg.inv(phiTphi), np.transpose(phiMat))
    return np.dot(MPMat, t)
