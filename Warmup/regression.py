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
    #design matrix given by phi_nj = phi_j(x_n)
    print x, t
    phiMat = basisPoly(x, M)
    phiTphi = np.dot(np.transpose(phiMat), phiMat)
    MPMat = np.dot(np.linalg.inv(phiTphi), np.transpose(phiMat))
    return np.dot(MPMat, t)
