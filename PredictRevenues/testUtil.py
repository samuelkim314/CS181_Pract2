import cPickle

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