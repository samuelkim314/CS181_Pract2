import numpy as np
import math

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

def basisFourier(x, M, T=0):
    Mcos = math.trunc((M - 1) / 2)
    Msin = M - 1 - Mcos
    msCos = np.arange(Mcos) + 1
    msSin = np.arange(Msin) + 1

    #period
    if T==0:
        T = np.amax(x) - np.amin(x)

    """if type(x) is float:
        sinSub = np.sin(x * msSin * 2 * np.pi / T)
        cosSub = np.cos(x * msCos * 2 * np.pi / T)
        return np.concatenate((np.array([1]), sinSub, cosSub), axis=0)
    else:"""
    ones = np.ones((len(x), 1))
    sinSub = np.sin(np.outer(x, msSin) * 2 * np.pi / T)
    cosSub = np.cos(np.outer(x, msCos) * 2 * np.pi / T)
    return np.concatenate((ones, sinSub, cosSub), axis=1)

def errorFun(x, t, w, basis):
    return 0.5*np.sum((t - np.dot(w, basis(x)))**2)

def maximum_likelihood(x, t, basis, M, beta = 0.5):
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
    return np.dot(MPMat, t), [1/beta for x in range(len(x))]

def bayesian_linear_regression(x, t, basis, M, alpha = 1, beta = 0.5):
    """Solves exactly for w, the weights. Not recommended for large data sets.
    Arguments:
        x:     array of inputs
        t:     array of corresponding target variables
        basis: function that calculates basis functions for inputs
        M:     number of basis functions to use
        alpha: initial prior distribution pararmeter
        beta:  precision, inverse variance
    Return:
        w:    array of parameters for the basis functions"""

    #design matrix
    Phi = basis(x, M)

    S_N = np.linalg.inv(alpha*np.identity(M) + beta*np.dot(np.transpose(Phi), Phi))
    m_N = beta*S_N.dot(np.transpose(Phi)).dot(t)

    return m_N, 1/beta + Phi[0].dot(S_N).dot(np.transpose(Phi))