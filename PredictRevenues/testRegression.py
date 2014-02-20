try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import random

def extract_feats_split(ffs, datafile="train.xml", global_feat_dict=None,
                        withhold=0):
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

    #Combine the 3 lists into 1 list to match the indices
    data = zip(fds, targets, ids)
    #Shuffles the data and divides them into training and testing data
    random.shuffle(data)
    trainData = data[:(len(data) - withhold)]
    testData = data[(len(data) - withhold):]
    #Divides the list into its components
    fds, targets, ids = zip(*trainData)
    fdsTest, targetsTest, idsTest = zip(*testData)

    X,feat_dict = make_design_mat(fds,global_feat_dict)
    XTest,_ = make_design_mat(fdsTest, feat_dict)

    return X, feat_dict, np.array(targets), ids, XTest, np.array(targetsTest), idsTest

def testMAE(preds, trues):
    return np.sum(np.absolute(preds - trues)) / len(preds)

## The following function does the feature extraction, learning, and prediction
def mainTest(withhold = 0):
    trainfile = "train.xml"
    #testfile = "testcases.xml"
    outputfile = "mypredictions.csv"  # feel free to change this or take it as an argument

    # TODO put the names of the feature functions you've defined above in this list
    ffs = [metadata_feats, unigram_feats]

    # extract features
    print "extracting training features..."
    X_train,global_feat_dict,y_train,train_ids, XTest,targetsTest,idsTest = extract_feats(ffs, trainfile, withhold=withhold)
    print "done extracting training features"
    print

    # TODO train here, and return regression parameters
    print "learning..."
    learned_w = splinalg.lsqr(X_train,y_train)[0]
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
    preds = XTest.dot(learned_w)
    print "done making predictions"
    print

    print "MAE on withheld data:", testMAE(preds, targetsTest)