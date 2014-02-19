import numpy as np

def basisPoly(x, M):
    return np.array([x**i for i in range(M)])

def errorFun(x, t, w, basis):
    return 0.5*np.sum((t - np.dot(w, basis(x)))**2)

def solveW(x, t, basis, M):
    #design matrix given by phi_nj = phi_j(x_n)
    desMat = np.transpose(basisPoly(x, M))
    MPMat = np.dot(np.linalg.inv(np.dot(np.transpose(desMat), desMat)), np.transpose(desMat))
    return np.dot(MPMat, t)
