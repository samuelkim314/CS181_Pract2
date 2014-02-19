import csv
import sys
import itertools
import numpy as np

def importWarmupData(filename="motorcycle.csv"):
    """Imports the motorcycle data from the csv file
    Arguments:
        filename (optional):     file name of the data
    Returns:
        dict with 'time': numpy array of times
                'force': numpy array of force at corresponding times
    """
    reader = csv.reader(open(filename, 'rb'))
    time = []
    force = []
    for row in itertools.islice(reader,1,None):
        time.append(float(row[0]))
        force.append(float(row[1]))

    return {"time": np.array(time), "force": np.array(force)}

def importTestData(n=5):
    time = np.arange(n)
    force = np.power(time, 2)

    return {"time": np.array(time), "force": np.array(force)}