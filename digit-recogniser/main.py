#!/usr/bin/env python3

import numpy as np
from sklearn import svm

def main():
    train = np.loadtxt(open("data/short_train.csv", "r"), delimiter=",", skiprows=1)
    trainResults = train[:,0]
    trainData = train[:,1:]
    clf = svm.SVC()
    x = clf(trainData, trainResults)
    print(x)


if __name__ == "__main__":
    main()
