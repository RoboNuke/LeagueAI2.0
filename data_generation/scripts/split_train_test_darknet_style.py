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
width = 960
height = 544

def unpack(argv):
    global trainSetSize, inputFile, outputFile, outConfig, configFile, datasetName, topSize, width, height
    try:
        opts, args = getopt.getopt(argv, "hi:c:o:s:k:n:t:w:h:", ["inputFile=", "configFile=", "outputFile=", "trainingSetSize=", "outputConfigFile=", "datasetName=", "top=", "width=", "heigt="])
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
        elif opt in ("-i","inputFile="):
            inputFile = arg
        elif opt in ("-k", "outputConfigFile="):
            outConfig = arg
        elif opt in ("-n", "datasetName"):
            datasetName = arg
        elif opt in ("-t", "top="):
            topSize = int(arg)
        elif opt in ("-w", "widthp="):
            width = int(arg)
        elif opt in ("-h", "height="):
            height = int(arg)
        elif opt in ("-s", "trainingSetSize="):
            trainSetSize=int(arg)
    labels = yaml.load(open(configFile))
    numClasses = len(labels["champions"]) + len(labels["creatures"])
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Variable", "Value"))
    print()
    print("{:25s}: {:2.3f}".format("Number of Classes", numClasses))
    print("{:25s}: {:2.3f}".format("Training Set Size", trainSetSize))
    print("{:25s}: {:25s}".format("Input File", inputFile))
    print("{:25s}: {:25s}".format("Location of outputs", outputFile))
    print("{:25s}: {:25s}".format("Location of output configuration file", outConfig))
    print("{:25s}: {:25s}".format("Dataset Name", datasetName))
    print("{:25s}: {:2.3f}".format("Top number to look at", topSize))
    print("{:25s}: {:2.3f}".format("Image Width", width))
    print("{:25s}: {:2.3f}".format("Image Height", height))
    
    print("----------------------------------------------------------")


def resize(img):
    global width, height
    return cv2.resize(width, height)

def splitData(dataset):
    global trainSetSize
    random.shuffle(dataset)
    train = dataset[:trainSetSize]
    test = dataset[trainSetSize:]
    return train, test

def copyData(trainFiles, testFiles):
    if not os.path.exists(outputFile + "test/"):
        os.makedirs(outputFile + "test/")
    if not os.path.exists(outputFile + "train/"):
        os.makedirs(outputFile + "train/")
    newTrain = []
    newTest = []

    os.system('cp ' + inputFile + "labels.txt " + outputFile + datasetName + ".names")
    for f in trainFiles:
        k = f[5:-4]
        os.system('cp ' + inputFile + "images/" + f + " " + outputFile + "train/" + datasetName + k + ".jpg")
        os.system('cp ' + inputFile + "labels/label" + k + ".txt " + outputFile + "train/" + datasetName + k + ".txt")
        newTrain.append(outputFile + "train/" + datasetName + k + ".jpg")
    for f in testFiles:
        k = f[5:-4]
        os.system('cp ' + inputFile + "images/" + f + " " + outputFile + "test/" + datasetName + k + ".jpg")
        os.system('cp ' + inputFile + "labels/label" + k + ".txt " + outputFile + "test/" + datasetName + k + ".txt")
        newTest.append(outputFile + "test/" + datasetName + k + ".jpg")
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
    global outConfig, topSize, datasetName, numClasses, outputFile
    with open(outConfig + datasetName + ".data", "w") as f:
        f.write("classes=" + str(numClasses) + "\n")
        f.write("train = " + outputFile + "train.list\n")
        f.write("valid = " + outputFile + "test.list\n")
        f.write("labels = " + outputFile + datasetName + ".name\n")
        f.write("backup = backup/" + datasetName+ "\n")
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
    
