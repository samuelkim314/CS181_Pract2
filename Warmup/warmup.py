import utils
import regression

data = utils.importWarmupData()
#data = utils.importTestData()

x = data['time']
t = data['force']
basis = regression.basisPoly

w = regression.solveW(x, t, basis, 10)

utils.plotRegression(x, t, w, basis)