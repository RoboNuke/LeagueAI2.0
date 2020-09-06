from os import listdir
import numpy as np
import random

# This script is used to split the datasets generated with bootstrap.py into test and training dataset

# Attention if you use darknet to train: the structure has to be exactly as follows:
# - Dataset
# -- images
# --- XYZ0.jpg
# --- XYZ1.jpg
# -- labels
# --- XYZ0.txt
# --- XYZ1.txt
# -- train.txt
# -- test.txt


datasetSize = 10
trainSetSize=2
inputFile="../data/3data_set/"
outputFile="/home/hunter/LeagueAI2.0/data_generation/data/4darknet_ready/"
outConfig="/home/hunter/LeagueAI2.0/data_generation/cfg/testSet.cfg"
configFile="../cfg/vayne.yaml"
topSize=2


def splitData(dataset):
    global trainSetSize
    random.shuffle(dataset)
    train = dataset[:trainSetSize]
    test = dataset[trainSetSize:]
    return train, test

def copyData(trainFiles, testFiles):
    if not os.path.exists(outputFile + "test/labels"):
        os.makedirs(outputFile + "test/labels")
    if not os.path.exists(outputFile + "train/labels"):
        os.makedirs(outputFile + "train/labels")
    if not os.path.exists(outputFile + "test/images"):
        os.makedirs(outputFile + "test/images")
    if not os.path.exists(outputFile + "train/images"):
        os.makedirs(outputFile + "train/images")
    newTrain = []
    newTest = []
    for f in trainFiles:
        k = f[-4]
        os.system('cp ' + f + " " + outputFile + "train/images/image" + k + ".jpg")
        os.system('cp ' + inputFile + "labels/label" + k + ".txt " + outputFile + "train/labels/label" + k + ".txt")
        newTrain.append(outputFile + "train/images/image" + k + ".jpg")
    for f in testFiles:
        k = f[-4]
        os.system('cp ' + f + " " + outputFile + "test/images/image" + k + ".jpg")
        os.system('cp ' + inputFile + "labels/label" + k + ".txt " + outputFile + "test/labels/label" + k + ".txt")
        newTest.append(outputFile + "test/images/image" + k + ".jpg")
    return newTrain, newTest
        
        
    
def makeListFile(trainFiles, testFiles):
    global outputFile
    if not os.path.exists(outputFile + "test.list"):
        os.makedirs(outputFile + "test.list")
    if not os.path.exists(outputFile + "train.list"):
        os.makedirs(outputFile + "train.list")
    with open(outputFile + "test.list", "w") as f:
        for fil in testFiles::
            f.write(testFiles + "\n")
    with open(outputFile + "train.list", "w") as f:
        for fil in trainFiles:
            f.write(fil+"\n")

def makeLabelFile(path):
    global outConfig, topSize
    if not os.path.exists(outConfig):
        os.makedirs(outConfig)
    with open(outConfig, "w") as f:
        f.write("classes=" + numClasses + "\n")
        f.write("train = " + outputFile + "train.list\n")
        f.write("valid = " + outputFile + "test.list\n")
        f.write("labels = " + outputFile + "labels.txt\n")
        f.write("backup = backup/\n")
        f.write("top=" + str(topSize) + "\n")

if __name__ == "__main__":
    
