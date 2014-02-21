import utils
import regression

data = utils.importWarmupData()
#data = utils.importTestData()

x = data['time']
t = data['force']
#basis = regression.basisPoly
basis = regression.basisFourier

w = regression.solveW(x, t, basis, 10)

utils.plotRegression(x, t, w, basis)