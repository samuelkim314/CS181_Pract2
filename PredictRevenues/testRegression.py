import regression_starter

#total of 1147 movies in the training set
#set this to 0 to predict on the test set
withhold = 100
extractedFile = None
trainFile = None
testFile = None

"""values of 'loadFile':
  'extract': load the extracted features
              use if feature-functions are unchanged
              'extractedFile' must be provided
  'split'  : load the partitioned data
              'trainFile' and 'testFile' must be provided
  None     : Load all data from original xml files"""
params = {'withhold': 100,
          'load': None,
          'extractedFile': 'extracted2',
          'trainFile': None,
          'testFile': None
          }

regression_starter.mainTest(withhold)