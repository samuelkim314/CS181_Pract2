import regression_starter

#total of 1147 movies in the training set
#set this to 0 to predict on the test set
withhold = 100

"""values of 'loadFile':
  'extract': load the extracted features
              use if feature-functions are unchanged
              'extractedFile' must be provided
  'split'  : load the partitioned data
              'trainFile' and 'testFile' must be provided
  None     : Load all data from original xml files"""
params = {'withhold': 100,
          'load': 'extract',
          'extractFile': 'featW100',
          'trainFile': None,
          'testFile': None,
          'writePredict': False,
          'outputFile': 'predictions.csv'
          }

regression_starter.mainTest(withhold, params)