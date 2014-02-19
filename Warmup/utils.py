import csv
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt

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

def plotRegression(x, t, w, basis):
    #setting up axis limits
    #TODO: allow input axis limits
    rangeX = np.amax(x) - np.amin(x)
    minX = np.amin(x) - rangeX / 10.0
    maxX = np.amax(x) + rangeX / 10.0
    rangeY = np.amax(t) - np.amin(t)
    minT = np.amin(t) - rangeY / 10.0
    maxT = np.amax(t) + rangeY / 10.0

    #plot the data
    plt.plot(x, t, 'ro')
    plt.axis([minX, maxX, minT, maxT])

    #plot the predicted curve
    sampX = np.linspace(minX, maxX, 50)
    phiMat = basis(sampX, len(w))
    sampT = np.dot(phiMat, w)
    print sampX.size, sampT.size
    plt.plot(sampX, np.dot(phiMat, w))

    plt.show()