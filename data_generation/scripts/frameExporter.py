# This script just needs Opencv to run
# Install opencv using pip: python -m pip install opencv-python
# See this link for more information: https://www.scivision.co/install-opencv-python-windows/
import cv2
import numpy as np
import math
import sys, getopt
import os
import yaml
import time

outputFile = "../data/2cropped_images/"
inputFile = "../data/1videos/"
configFile = "../cfg/vayne.yaml"
skipFrames = 40
inFilePrefix = "test_"
outFilePrefix = "2test"

# these are for the background
# to-do put these in a config file or something so code doesn't have to open idk
DEBUG = False
bgColor1 = (166, 81, 96)
bgVar = 25
topCut = 50
notChamp = 0
closeBox = 10

def unpack(argv):
    global outputFile, inputFile, configFile, skipFrames, inFilePrefix, outFilePrefix, DEBUG

    try:
        opts, args = getopt.getopt(argv,"ho:i:c:s:p:q:d", ["outputFile=","inputFile=", "configFile=", "skipFrames=","outFilePrefix=","inFilePrefix=", "debug="])
    except getopt.GetoptError:
        print('frameExporter.py -c <configuration file> -o <output file> -i <input file> -s <frames to skip> -p <output prefix> -q <input prefix> -d <True or False>')
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            print('frameExporter.py -c <configuration file> -o <output file> -i <input file> -s <frames to skip> -p <output prefix> -q <input prefix>')
            sys.exit()
        elif opt in ("-c", "configFile="):
            configFile = arg
        elif opt in ("-o","outputFile="):
            outputFile = arg
        elif opt in ("-p","outFilePrefix="):
            outFilePrefix = arg
        elif opt in ("-i","inputFile="):
            inputFile = arg
        elif opt in ("-s","skipFrames="):
            skipFrames = int(arg)
        elif opt in ("-q", "inFilePrefix="):
            inFilePrefix = arg
        elif opt in ("-d", "debug="):
            DEBUG = True
    todoList = yaml.load(open(configFile))
    champs = todoList["champions"]
    creeps = todoList["creatures"]
    if DEBUG:
        print("Working in Debug Mode!")
    print("Will load the following champions' video:")
    for c in champs:
        print("--", c)
    print("Will load the following creatures' video:")
    for c in creeps:
        print("--", c)
    print("Their exported frames can be found at: ", outputFile)
    print("All exported frames will be of the form: ", outFilePrefix, "/[champion/creature name].png")
    print("Videos will be loaded from: ", inputFile)
    print("Will load videos of the form: ", inFilePrefix, "\b[champion/creature name].avi")
    print("Will skip ", skipFrames, " frames.")
    return (champs, creeps)

def exportVid(vid, champName):
    global outputFile, inputFile, configFile, skipFrames, inFilePrefix, outFilePrefix
    if vid.isOpened() == False:
        print("Failure to load video")
    else:
        outDir = outputFile + outFilePrefix + "/" + champName 
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        
        frameCount = 0
        outputCounter = 0
        while(vid.isOpened()):
            if DEBUG and outputCounter == 4:
                break
            ret, frame = vid.read()
            if ret:
                if frameCount % skipFrames == 0:
                    #write to file
                    outputCounter = outputCounter + 1
                    frame = frame[topCut:,:]
                    mask = createMask(frame)
                    frame = cv2.bitwise_and(frame, mask)
                    (frame, mask) = cropDown(frame, mask)
                    
                    cv2.imwrite(outDir + "/" + champName+ str(outputCounter) + ".jpg", frame)
                    cv2.imwrite(outDir + "/" + champName+ str(outputCounter) + "_mask.jpg", mask)
                frameCount = frameCount + 1
            else:
                print('Processed', champName, "\b's video")
                break
        return outputCounter
    return -1
                
def createMask(frame):
    """ Creates the bounding box and a mask """
    global notChamp
    # convert to bgra
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    mask = np.zeros((frame.shape[0],frame.shape[1],3),np.uint8)
    #bgGone = np.where(frame==bgColor1,[0,0,0],[255,255,255])
    for u in range(frame.shape[0]):
        for v in range(frame.shape[1]):
            if ((frame.item(u,v,0) >= bgColor1[0] - bgVar and frame.item(u,v,0) < bgColor1[0] + bgVar) and
                 (frame.item(u,v,1) > bgColor1[1] - bgVar and frame.item(u,v,1) < bgColor1[1] + bgVar) and
                   (frame.item(u,v,2) > bgColor1[2] - bgVar and frame.item(u,v,2) < bgColor1[2] + bgVar)):
                mask.itemset((u,v,0),0)
                mask.itemset((u,v,1),0)
                mask.itemset((u,v,2),0)
            else:
                mask.itemset((u,v,0),255)
                mask.itemset((u,v,1),255)
                mask.itemset((u,v,2),255)
    kernel = np.ones((closeBox,closeBox),np.uint8)
    kernel2 = np.ones((int(closeBox/2), int(closeBox/2)), np.uint8)
    #cv2.imshow("Mask", mask)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.erode(mask,kernel2,iterations = 1)
    #mask = cv2.dilate(mask,kernel2,iterations = 1)
    return mask

def cropDown(frame, mask):
    global notChamp
    mX = 0
    mY = 0
    nX = frame.shape[0]-1
    nY = frame.shape[1]-1
    for v in range(frame.shape[1]):
        for u in range(frame.shape[0]):
            if frame.item(u,v,0) != notChamp and frame.item(u,v,1) != notChamp and frame.item(u,v,2) != notChamp:
                mY = v
    for u in range(frame.shape[0]):
        for v in range(frame.shape[1]):
            if frame.item(u,v,0) != notChamp and frame.item(u,v,1) != notChamp and frame.item(u,v,2) != notChamp:
                mX = u
    for v in range(frame.shape[1]-1,0,-1):
        for u in range(frame.shape[0]-1,0,-1):
            if frame.item(u,v,0) != notChamp and frame.item(u,v,1) != notChamp and frame.item(u,v,2) != notChamp:
                nY = v
    for u in range(frame.shape[0]-1,0,-1):
        for v in range(frame.shape[1]-1,0,-1):
            if frame.item(u,v,0) != notChamp and frame.item(u,v,1) != notChamp and frame.item(u,v,2) != notChamp:
                nX = u
    return (frame[nX:mX,nY:mY], mask[nX:mX,nY:mY])
    
                          
    
if __name__ == "__main__":
    """
    # tests for mask & bbox generation
    frame = cv2.imread("../data/2cropped_images/test_Summoners Rift Chaos Minion Melee/test_Summoners Rift Chaos Minion Melee1.jpg")
    frame = frame[topCut:,:]
    t = time.clock()
    mask = createMask(frame)
    #mask = cv2.resize(mask, frame.shape[1::-1])
    frame = cv2.bitwise_and(frame, mask)
    cropFrame = cropDown(frame)
    
    tt = time.clock()
    print "Time:", tt - t
    cv2.imshow("Initial Image",frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Cropped Image", np.array(cropFrame, dtype=np.uint8))
    cv2.waitKey(0)
    sys.exit()
    
    """
    (champs, creeps) = unpack(sys.argv[1:])

    frameCountDic = {}
                            
    for champ in champs:
        print( "Creating", champ, "\b's frames")
        print(inputFile + inFilePrefix + champ + ".avi")
        vid = cv2.VideoCapture(inputFile + inFilePrefix + champ + ".avi")
        frameCountDic[champ] = exportVid(vid, champ)
        

    for creep  in creeps:
        print( "Creating", creep, "\b's frames")
        vid = cv2.VideoCapture(inputFile + inFilePrefix + creep + ".avi")
        frameCountDic[creep] = exportVid(vid, creep)

    #print frameCountDic
    countPath = outputFile + outFilePrefix
    if not os.path.exists(countPath):
        os.makedirs(countPath)
    with open(countPath + "/count.yaml",'w') as file:
        yaml.dump(frameCountDic, file)
    sys.exit()

      

