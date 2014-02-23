import cPickle
import random
import regression_starter as regress
import numpy as np

def unpickle(file):
    """Loads and returns a pickled data structure in the given `file` name
    Example usage:
        data = unpickle('output/U_20_std')
    """
    fo = open(file, 'rb')
    data = cPickle.load(fo)
    fo.close()
    return data

def pickle(data, file):
    """Dumps data to a file
    Example usage:
        pickle(U, 'output/U_20_std')
    """
    fo = open(file,'wb')
    cPickle.dump(data,fo)
    fo.close()

def splitData(fds, targets, ids, withhold=0):
    #Combine the 3 lists into 1 list to match the indices
    data = zip(fds, targets, ids)
    #Shuffles the data and divides them into training and testing data
    random.shuffle(data)
    trainData = data[:(len(data) - withhold)]
    testData = data[(len(data) - withhold):]
    #Divides the list into its components
    fds, targets, ids = zip(*trainData)
    fdsTest, targetsTest, idsTest = zip(*testData)

    return fds, targets, ids, fdsTest, targetsTest, idsTest

def loadData(params, withhold, ffs, trainfile="train.xml", testfile="testcases.xml"):
    """
    loads the movie data

    arguments:
        params      : dict with several keys:
            load        : loading mode; either: 'extract' to load from 
                        `params['extractFile']`, 'split' to load from 
                        `params['splitFile']`, or None to extract features and 
                        save to `params['extractFile']` and/or 
                        `params['splitFile']`. 
            extractFile : file to load/save extracted features to/from,
                        depending on loading mode
            splitFile   : file to load/save split data to/from,
                        depending on loading mode
        withhold    : number of data points to withhold for cross-validation
        ffs         : list of feature functions
        trainfile   : path to training file (train.xml)
        testfile    : path to test cases file

    returns:

    """

    # extract and split the data anew
    if params['load']==None:
        fds, targets, train_ids = regress.extract_feats_helper(ffs, trainfile)
        print "loaded %d fds" % len(fds) 
        pickle((fds,targets,train_ids),params['extractFile'])

        if withhold==0:
            X_train,feat_dict = regress.make_design_mat(fds)
            y_train=np.array(targets)
            X_test,_,y_test,test_ids = regress.extract_feats(ffs, testfile, global_feat_dict=feat_dict)
        else:
            print "withholding %d of %d fds" % (withhold, len(fds)-withhold)
            fds, targets, train_ids, fdsTest, targetsTest, test_ids = splitData(fds, targets, train_ids, withhold)
            X_train,feat_dict = regress.make_design_mat(fds)
            X_test,_ = regress.make_design_mat(fdsTest, feat_dict)
            y_train=np.array(targets)
            y_test=np.array(targetsTest)
        if params['splitFile'] != None:
            pickle((X_train, y_train, train_ids,X_test,y_test,test_ids), params['splitFile'])
    
    # load data from `params['extractFile']`, but split it anew
    elif params['load']=='extract':
        fds,targets,ids=unpickle(params['extractFile'])
        print "loaded %d fds" % len(fds) 
        print "withholding %d of %d fds" % (withhold, len(fds)-withhold)
        fds, targets, train_ids, fdsTest, targetsTest, test_ids = splitData(fds, targets, ids, withhold)
        X_train,feat_dict = regress.make_design_mat(fds)
        X_test,_ = regress.make_design_mat(fdsTest, feat_dict)
        y_train=np.array(targets)
        y_test=np.array(targetsTest)
        if params['splitFile'] != None:
            pickle((X_train, y_train, train_ids,X_test,y_test,test_ids), params['splitFile'])

    # load data from `params['splitFile']`
    elif params['load']=='split':
        X_train, y_train, train_ids,X_test,y_test,test_ids = unpickle(params['splitFile'])
        print "loaded %d fds" % len(train_ids) 
        print "withholding %d of %d fds" % (len(test_ids), len(train_ids))

    return X_train, y_train, train_ids,X_test,y_test,test_ids