import os, sys, getopt
import numpy as np
import random
import yaml

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


trainSetSize=8
inputFile="../data/3data_set/"
outputFile="/home/hunter/LeagueAI2.0/data_generation/data/4darknet_ready/"
outConfig="/home/hunter/LeagueAI2.0/data_generation/cfg/"
configFile="../cfg/vayne.yaml"
datasetName = "vayneTest"
topSize=2
numClasses = 2


def unpack(argv):
    global trainSetSize, inputFile, outputFile, outConfig, configFile, datasetName, topSize
    try:
        print(argv)
        opts, args = getopt.getopt(argv, "hi:c:o:s:k:n:t:", ["inputFile=", "configFile=", "outputFile=", "trainingSetSize=", "outputConfigFile=", "datasetName=", "top="])
    except getopt.GetoptError:
        print(getopt.GetoptError)
        print("split_train_test_darknet_style.py -i <input file> -c <configuration file> -o <output file> -s <desired training set size> -k <output cfg file> -n <dataset Name> -t <top n>")
        sys.exit()

    for opt, arg in opts:
        if opt == "-h":
            print("split_train_test_darknet_style.py -i <input file> -c <configuration file> -o <output file> -s <desired training set size> -k <output cfg file> -n <dataset Name> -t <top n>")
            sys.exit()
        elif opt in ("-c", "configFile="):
            configFile = arg
        elif opt in ("-o","outputFile="):
            outputFile = arg
        elif opt in ("-s","trainingSetSize="):
            outFilePrefix = int(arg)
        elif opt in ("-i","inputFile="):
            inputFile = arg
        elif opt in ("-k", "outputConfigFile="):
            outConfig = arg
        elif opt in ("-n", "datasetName"):
            datasetName = arg
        elif opt in ("-t", "top="):
            topSize = int(arg)
    labels = yaml.load(open(configFile))
    numClasses = len(labels["champions"]) + len(labels["creatures"])
            
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

    os.system('cp ' + inputFile + "labels.txt " + outputFile + "labels.txt")
    for f in trainFiles:
        k = f[5:-4]
        os.system('cp ' + inputFile + "images/" + f + " " + outputFile + "train/images/" + datasetName + k + ".jpg")
        os.system('cp ' + inputFile + "labels/label" + k + ".txt " + outputFile + "train/labels/" + datasetName + k + ".txt")
        newTrain.append(outputFile + "train/images/" + datasetName + k + ".jpg")
    for f in testFiles:
        k = f[5:-4]
        os.system('cp ' + inputFile + "images/" + f + " " + outputFile + "test/images/" + datasetName + k + ".jpg")
        os.system('cp ' + inputFile + "labels/label" + k + ".txt " + outputFile + "test/labels/" + datasetName + k + ".txt")
        newTest.append(outputFile + "test/images/" + datasetName + k + ".jpg")
    print("Completed copying files to new location")
    return newTrain, newTest
    
def makeListFile(trainFiles, testFiles):
    global outputFile
    if not os.path.exists(outputFile):
        os.makedirs(outputFile)
    with open(outputFile + "test.list", "w") as f:
        for fil in testFiles:
            f.write(fil + "\n")
    with open(outputFile + "train.list", "w") as f:
        for fil in trainFiles:
            f.write(fil+"\n")
    print("List Files Made")

def makeLabelFile():
    global outConfig, topSize, datasetName
    with open(outConfig + datasetName + ".cfg", "w") as f:
        f.write("classes=" + str(numClasses) + "\n")
        f.write("train = " + outputFile + "train.list\n")
        f.write("valid = " + outputFile + "test.list\n")
        f.write("labels = " + outputFile + "labels.txt\n")
        f.write("backup = backup/\n")
        f.write("top=" + str(topSize) + "\n")
    print("Label File Made")

if __name__ == "__main__":
    print("\nStarting to split dataset and copy to final locations")
    # get a list of all the files in the training set
    unpack(sys.argv[1:])
    dataset = os.listdir(inputFile + "images/")
    train, test = splitData(dataset)
    train, test = copyData(train, test)
    makeListFile(train, test)
    makeLabelFile()
    
