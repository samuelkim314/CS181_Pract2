import regression_starter

#total of 1147 movies in the training set
#set this to 0 to predict on the test set
withhold = 100
extractedFile = None
trainFile = None
testFile = None

params = {'withhold': 0,
          'extractFeatures': True,
          'extractedFile': None,
          'loadPartition': False,
          'trainFile': None,
          'testFile': None
          }

regression_starter.mainTest(withhold)