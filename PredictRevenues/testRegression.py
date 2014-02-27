import regression_starter

#total of 1147 movies in the training set
#set this to 0 to predict on the test set
withhold = 100

"""values of 'load':
  'extract': load the extracted features; use if feature-functions are unchanged
              'extractedFile' must be provided
  'split'  : load the partitioned data
              'splitFile' must be provided
  None     : Load all data from original xml files
              if 'extractFile' provided, will save the extracted features
              if 'splitfile' provided, will also save the split data

    value of splitMethod:
    0: random
    1: last 'withhold' are withheld
    2: first 'withhold' are withheld"""
"""loadParams = {'load': 'extract',
          'extractFile': 'extracted2ffs',
          'splitFile': 'withhold100',
          'writePredict': True,
          'outputFile': 'predictDamp15k.csv'
          'splitMethod': 1
          }"""
loadParams = {'load': None,
          'extractFile': None,
          'splitFile': None,
          'writePredict': False,
          'outputFile': 'predictBigram.csv',
          'splitMethod': 2
          }

regression_starter.mainTest(withhold, loadParams)