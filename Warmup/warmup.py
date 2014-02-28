import utils
import regression

data = utils.importWarmupData()
#data = utils.importTestData()

x = data['time']
t = data['force']
#basis = regression.basisPoly
basis = regression.basisFourier
M = 16

'''Maximum likelihood estimation'''
w, var = regression.maximum_likelihood(x, t, basis, M)
print w

#utils.plt.figure()
utils.plotRegression(x, t, w, basis, var, 'blue')
utils.plt.savefig("maximum_likelihood_estimation_16")

'''Bayesian linear regression'''
w, pred = regression.bayesian_linear_regression(x, t, basis, M)
print w

utils.plotRegression(x, t, w, basis, pred, 'green')
utils.plt.savefig("bayesian_linear_regression_16")


utils.plt.show()