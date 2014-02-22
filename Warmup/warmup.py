import utils
import regression

data = utils.importWarmupData()
#data = utils.importTestData()

x = data['time']
t = data['force']
#basis = regression.basisPoly
basis = regression.basisFourier

'''Maximum likelihood estimation'''
w = regression.maximum_likelihood(x, t, basis, 8)
print w

utils.plotRegression(x, t, w, basis)

'''Bayesian linear regression'''
w = regression.bayesian_linear_regression(x, t, basis, 8)
print w

utils.plotRegression(x, t, w, basis)