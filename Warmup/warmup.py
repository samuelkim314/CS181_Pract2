import utils
import regression

#data = utils.importWarmupData()
data = utils.importTestData()
print regression.solveW(data['time'], data['force'], regression.basisPoly, 10)

print regression.basisPoly(5, 5)