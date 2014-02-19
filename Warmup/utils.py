import csv
import sys
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
    for row in reader:
        time.append(row[0])
        force.append(row[1])

    return {"time": np.array(time[1:]), "force": np.array(force[1:])}