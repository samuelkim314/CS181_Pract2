## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up:
## ----------------
## main() will run code to extract features, learn, and make predictions.
##
## extract_feats() is called by main(), and it will iterate through the
## train/test xml files and extract each instance into a util.MovieData object.
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each util.MovieData object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code for naive linear regression and prediction so you
## have a sense of where/what to modify.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take a util.MovieData object representing
## a single movie, and return a dictionary mapping feature names to their respective
## numeric values.
## For instance, a simple feature-function might map a movie object to the
## dictionary {'company-Fox Searchlight Pictures': 1}. This is a boolean feature
## indicating whether the production company of this move is Fox Searchlight Pictures,
## but of course real-valued features can also be defined. Because this feature-function
## will be run over MovieData objects for each movie instance, we will have the (different)
## feature values of this feature for each movie, and these values will make up
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions will be unioned
## so we can collect all the feature values associated with a particular instance.
##
## Two example feature-functions, metadata_feats() and unigram_feats() are defined
## below. These extract metadata and unigram text features, respectively.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.


from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import util
import testUtil as test
import time

def extract_feats(ffs, datafile="train.xml", global_feat_dict=None):
    fds, targets, ids = extract_feats_helper(ffs, datafile)

    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, np.array(targets), ids

def extract_feats_helper(ffs, datafile="train.xml"):
    """
    arguments:
      ffs are a list of feature-functions.
      datafile is an xml file (expected to be train.xml or testcases.xml).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that
      the columns of the test matrix align correctly.

    returns:
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target values, and a list of movie ids in order of their
      rows in the design matrix
    """
    fds = [] # list of feature dicts
    targets = []
    ids = []
    begin_tag = "<instance" # for finding instances in the xml file
    end_tag = "</instance>"
    in_instance = False
    curr_inst = [] # holds lines of file associated with current instance

    # start iterating thru file
    with open(datafile) as f:
        # get rid of first two lines
        _ = f.readline()
        _ = f.readline()
        for line in f:
            if begin_tag in line:
                if in_instance:
                    assert False  # cannot have nested instances
                else:
                    curr_inst = [line]
                    in_instance = True
            elif end_tag in line:
                # we've read in an entire instance; we can extract features
                curr_inst.append(line)
                # concatenate the lines we've read and parse as an xml element
                movie_data = util.MovieData(ET.fromstring("".join(curr_inst)))
                rowfd = {}
                # union the output of all the feature functions over this instance
                [rowfd.update(ff(movie_data)) for ff in ffs]
                # add the final dictionary for this instance to our list
                fds.append(rowfd)
                # add target val
                targets.append(movie_data.target)
                # keep track of the movie id's for later
                ids.append(movie_data.id)
                # reset
                curr_inst = []
                in_instance = False
            elif in_instance:
                curr_inst.append(line)

    return fds, targets, ids

def extract_feats_split(ffs, datafile="train.xml", global_feat_dict=None,
                        withhold=0):
    fds, targets, ids = extract_feats_helper(ffs, datafile)

    fds, targets, ids, fdsTest, targetsTest, idsTest = test.splitData(fds, targets, ids, withhold)

    X,feat_dict = make_design_mat(fds,global_feat_dict)
    XTest,_ = make_design_mat(fdsTest, feat_dict)

    return X, feat_dict, np.array(targets), ids, XTest, np.array(targetsTest), idsTest

def testMAE(preds, trues):
    return np.sum(np.absolute(preds - trues)) / len(preds)

def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that
      the columns of the test matrix align correctly.

    returns:
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict

    cols = []
    rows = []
    data = []
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)


    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    return X, feat_dict

## Here are two example feature-functions. They each take in a util.MovieData
## object, and return a dictionary mapping feature-names to numeric values.
## TODO: modify these functions, and/or add new ones.
def metadata_feats(md):
    """
    arguments:
      md is a util.MovieData object
    returns:
      a dictionary containing a mapping from a subset of the possible metadata features
      to their values on this util.MovieData object
    """
    d = {}
    for k,v in md.__dict__.iteritems():
        if k in util.MovieData.implicit_list_atts or k in util.MovieData.reviewers:
            continue
        if k == "target":
            continue
        if isinstance(v, list):
            d.update([(k+"-"+val,1) for val in v])
        elif isinstance(v, float):
            d[k] = v
        elif isinstance(v, bool):
            d[k] = float(v)
        #elif k == "release_date":
        #    d[k] = numDate(v)
        #    y,m,day = numDate2(v)
        #    d[k+"_year"] = y
        #    d[k+"_month"]=m
        #    d[k+"_day"]=day
        else:
            d[k+"-"+v] = 1
    return d

def numDate(str):
    date = time.strptime(str, "%B %d, %Y")
    monFrac = (date.tm_mon - 1) / 12.0
    dayFrac = (date.tm_mday - 1) / 365.0
    return date.tm_year + monFrac + dayFrac

def numDate2(str):
    date = time.strptime(str, "%B %d, %Y")
    return date.tm_year, date.tm_mon, date.tm_mday

def unigram_feats(md):
    """
    arguments:
      md is a util.MovieData object
    returns:
      a dictionary containing a mapping from unigram features from the reviews
      to their values on this util.MovieData object
    """
    c = Counter()
    for rev in util.MovieData.reviewers:
        if hasattr(md,rev):
            # count occurrences of asciified, lowercase, non-numeric unigrams
            # after removing punctuation
            c.update(token for token in
                        util.punct_patt.sub("",
                         util.asciify(md.__dict__[rev].strip().lower())).split()
                          if util.non_numeric(token))
    return c

def bigram_feats(md):
    c = Counter()
    for rev in util.MovieData.reviewers:
        if hasattr(md,rev):
            # count occurrences of asciified, lowercase, non-numeric unigrams
            # after removing punctuation
            wordList = util.punct_patt.sub("",
                         util.asciify(md.__dict__[rev].strip().lower())).split()
            wordList = [x for x in wordList if util.non_numeric(x)]
            bigrams = zip(wordList, wordList[1:])
            c.update(token for token in bigrams)
    return c

def bigram_feats_noStop(md):
    c = Counter()
    for rev in util.MovieData.reviewers:
        if hasattr(md,rev):
            # count occurrences of asciified, lowercase, non-numeric unigrams
            # after removing punctuation
            stopWords = util.getStopWords()
            wordList = util.punct_patt.sub("",
                         util.asciify(md.__dict__[rev].strip().lower())).split()
            wordList = [x for x in wordList if util.non_numeric(x) and util.notStopWord(x, stopWords)]
            bigrams = zip(wordList, wordList[1:])
            c.update(token for token in bigrams)
    return c

def unigram_noStop(md):
    """
    arguments:
      md is a util.MovieData object
    returns:
      a dictionary containing a mapping from unigram features from the reviews
      to their values on this util.MovieData object, with stop words removed
    """
    unigramCount = unigram_feats(md)
    for sword in util.getStopWords():
        del unigramCount[sword]

    return unigramCount


## The following function does the feature extraction, learning, and prediction
def main(X_train=None, global_feat_dict=None):
    trainfile = "train.xml"
    testfile = "testcases.xml"
    outputfile = "mypredictions2.csv"  # feel free to change this or take it as an argument

    # TODO put the names of the feature functions you've defined above in this list
    ffs = [metadata_feats, unigram_feats]

    if X_train == None and global_feat_dict == None:
        # extract features
        print "extracting training features..."
        X_train,global_feat_dict,y_train,train_ids = extract_feats(ffs, trainfile)
        print "done extracting training features"
        print

    # TODO train here, and return regression parameters
    print "learning..."
    #learned_w = splinalg.lsqr(X_train,y_train)[0]
    learned_w = splinalg.lsmr(X_train,y_train)[0]
    print "done learning"
    print

    # get rid of training data and load test data
    del X_train
    del y_train
    del train_ids
    print "extracting test features..."
    X_test,_,y_ignore,test_ids = extract_feats(ffs, testfile, global_feat_dict=global_feat_dict)
    print "done extracting test features"
    print

    # TODO make predictions on text data and write them out
    print "making predictions..."
    preds = X_test.dot(learned_w)
    print "done making predictions"
    print

    print "writing predictions..."
    util.write_predictions(preds, test_ids, outputfile)
    print "done!"

def mainTest(withhold=0, params=None):

    #default value for params
    if params==None:
        params = {'withhold': 0,
          'load': None,
          'extractFile': None,
          'trainFile': None,
          'testFile': None,
          'writePredict': False,
          'outputFile': 'predictions.csv'
          }

    trainfile = "train.xml"
    testfile = "testcases.xml"

    # TODO put the names of the feature functions you've defined above in this list
    #ffs = [metadata_feats, unigram_feats]
    ffs = [metadata_feats, unigram_noStop]
    #ffs = [metadata_feats, bigram_feats_noStop]
    #ffs = [metadata_feats, bigram_feats_noStop, unigram_noStop]

    print "extracting training/testing features..."
    time1 = time.clock()
    X_train, y_train, train_ids,X_test,y_test,test_ids = test.loadData(params, withhold, ffs)
    time2 = time.clock()
    print "done extracting training/testing features", time2-time1, "s"
    print

    # TODO train here, and return regression parameters
    print "learning..."
    time1 = time.clock()
    #learned_w = splinalg.lsqr(X_train,y_train)[0]
    learned_w = splinalg.lsmr(X_train,y_train,damp=5000)[0]
    time2 = time.clock()
    print "done learning, ", time2-time1, "s"
    print

    # get rid of training data and load test data
    del X_train
    del y_train
    del train_ids

    # TODO make predictions on text data and write them out
    print "making predictions..."
    preds = X_test.dot(learned_w)
    print "done making predictions"
    print

    if withhold > 0:
        print "MAE on withheld data:", testMAE(preds, y_test)

    if params['writePredict']==True:
        print "writing predictions..."
        util.write_predictions(preds, test_ids, params['outputFile'])
        print "done!"


def mainTestIter(withhold=0, params=None):
    import learn

    #default value for params
    if params==None:
        params = {}

    params = dict({'withhold': 0,
      'load': None,
      'extractFile': None,
      'trainFile': None,
      'testFile': None,
      'writePredict': False,
      'outputFile': 'predictions.csv',

      # arguments to `learn`
      'options': {},

      # the option to cycle through
      'option': None,

      # range of values to cycle through
      'range': []
    }, **params)

    trainfile = "train.xml"
    testfile = "testcases.xml"

    # TODO put the names of the feature functions you've defined above in this list
    ffs = [metadata_feats, unigram_feats]

    print "extracting training/testing features..."
    time1 = time.clock()
    X_train, y_train, train_ids, X_test, y_test, test_ids = test.loadData(params, withhold, ffs)
    time2 = time.clock()
    print "done extracting training/testing features", time2-time1, "s"
    print

    # options for the learning engine
    options = params['options']

    # array to store MAEs for various values of learning options
    MAEs = []

    print "iterating over values of %s from %s ... %s" % (params['option'], params['range'][0], params['range'][-1])
    print "================================================================================"
    # iterate through each value of `params['option']` in `params['range']`
    # and calculate the MAE for that value
    for (i, value) in enumerate(params['range']):
        print "%s = %s" % (params['option'], str(value))
        op = dict(options)
        op[params['option']] = value
        decomp = None

        # train here, and return regression parameters
        print "learning..."
        time1 = time.clock()
        if 'reduction' in op and op['reduction'] != None:
            ((learned_w0, learned_w), decomp) = learn.learn(X_train, y_train, **op)
        else:
            (learned_w0, learned_w) = learn.learn(X_train, y_train, **op)

        time2 = time.clock()
        print "done learning, ", time2-time1, "s"
        print


        # make predictions on text data and write them out
        print "making predictions..."
        if decomp is None:
            preds = X_test.dot(learned_w) + learned_w0
        else:
            preds = decomp(X_test).dot(learned_w) + learned_w0
        print "done making predictions"
        print

        # cross-validate
        if withhold > 0:
            mae = testMAE(preds, y_test)
            print "MAE on withheld data: ", mae
            MAEs.append(mae)

        print "--------------------------------------------------------------------------------"

    print "================================================================================"

    # tabulate results
    results = dict()
    print "Results:"
    print "%s \t MAE" % params['option']
    for (i, value) in enumerate(params['range']):
        print "%s \t %d" % (value, MAEs[i])
        results[value] = MAEs[i]

    return results

if __name__ == "__main__":

    # Uncomment this to try different learning methods
    # mainTestIter(withhold=100,params={
    #   'load': 'extract',
    #   'extractFile': 'data/extracted2ffs',
    #   'outputFile': 'data/predictions.csv',
    #   'splitFile': 'data/splitFile',
    #   'writePredict': True,

    #   # Try different modes
    #   'options': { },
    #   'option': 'mode',
    #   'range': ['lsmr', 'lsqr', 'LinearRegression', 'ridge', 'ridgeCV', 'ElasticNetCV', 'LassoCV' ]
    #   # 'range': ['Perceptron', 'SGDRegressor' ]


    #   # # Try different decompositions
    #   # 'options': {
    #   #   'mode':'LarsCV',
    #   #   'n_components': 2
    #   # },
    #   # 'option': 'reduction',
    #   # 'range': ['tsvd', 'nmf']
    # })

    # ------------------------------------------------------------------------

    # Uncomment this to use Sam's code for cross-validation on a reasonable
    # subset of the data
    # mainTest(withhold=300,params={
    #   'withhold': 0,
    #   'load': 'extract',
    #   'extractFile': 'data/extracted2ffs',
    #   'outputFile': 'data/predictions.csv',
    #   'splitFile': 'data/splitFile',
    #   'writePredict': True
    #   })

    mainTest()
