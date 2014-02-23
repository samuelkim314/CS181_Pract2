import regression_starter

#total of 1147 movies in the training set
#set this to 0 to predict on the test set
withhold = 0

"""values of 'load':
  'extract': load the extracted features; use if feature-functions are unchanged
              'extractedFile' must be provided
  'split'  : load the partitioned data
              'splitFile' must be provided
  None     : Load all data from original xml files
              if 'extractFile' provided, will save the extracted features
              if 'splitfile' provided, will also save the split data"""
loadParams = {'load': 'extract',
          'extractFile': 'extracted2ffs',
          'splitFile': 'withhold100',
          'writePredict': True,
          'outputFile': 'predictPoly2.csv'
          }

regression_starter.mainTest(withhold, loadParams)