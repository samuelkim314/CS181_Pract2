import csv
import sys

def importWarmupData(filename="motorcycle.csv"):
    reader = csv.reader(open(filename, 'rb'))
    time = []
    force = []
    for row in reader:
        time.append(row[0])
        force.append(row[1])

    return {"time": time[1:], "force": force[1:]}