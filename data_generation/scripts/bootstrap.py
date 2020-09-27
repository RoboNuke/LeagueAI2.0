from __future__ import division
import sys, getopt
import os
import yaml

from random import random, choice, randint, uniform
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageColor, ImageDraw
####### Object Classes ##############
# 0 minion
# 1 melee
# 2 caster
# 3 canon
# 4 vayne


labelColor={ 0: ImageColor.getrgb("blue"),
             1: ImageColor.getrgb("red"),
             2: ImageColor.getrgb("green"),
             3: ImageColor.getrgb("purple")}
#Globals

configFile = "../cfg/vayne.yaml"
countFile = "count.yaml"
outputFile = "../data/3data_set/"
inputFile = "../data/2cropped_images/"
inFilePrefix = "test"
mapFile = "../data/0map/"
numImages = 11
labels = {}
DEBUG = False

# Probability Params
# champ caps
haveChamp = .95
maxChamp = 1

# creep caps
haveCreep = 1.0
maxCreep = 7
creepDist = [1.0]

# define distribution over possible rotations
maxRot = 10  # deg variance

# define how tightly packed the creeps are
creepB = 200
creepVar = 150 # pixels

# Scaling 
minScale = .35
maxScale = .5
creepScaleFactors = {}

mapImgs = []
width = 960
height = 544
padding = 4

def unpack(argv):
    global outputFile, inputFile, countFile, inFilePrefix, DEBUG, mapFile, numImages, configFile, labelFile, width, height, creepScaleFactors
    try:
        opts, args = getopt.getopt(argv,"hc:o:i:k:q:m:n:l:w:j:p:d", ["configFile=", "outputFile=","inputFile=", "countFile=","inFilePrefix=","mapFile=", "numberOfImages=","labelFile=", "debug=", "width=", "height=", "padding="])
    except getopt.GetoptError:
        print('frameExporter.py -c <config file> -k <count file> -o <output file> -i <input file> -q <input prefix> -l <label file> -d (debug)')
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            print('frameExporter.py -c <config file> -k <count file> -o <output file> -i <input file> -q <input prefix> -l <label file> -d (debug)')
            sys.exit()
        elif opt in ("-k", "countFile="):
            countFile = arg
        elif opt in ("-c", "configFile="):
            configFile = arg
        elif opt in ("-o","outputFile="):
            outputFile = arg
        elif opt in ("-i","inputFile="):
            inputFile = arg
        elif opt in ("-q", "inFilePrefix="):
            inFilePrefix = arg
        elif opt in ("-d", "debug="):
            DEBUG = True
        elif opt in ("-m", "mapFile="):
            mapFile = arg
        elif opt in ("-n", "numberOfImages="):
            numImages = int(arg)
        elif opt in ("-l", "labelFile="):
            labelFile = arg
        elif opt in ("-w", "width="):
            width = int(arg)
        elif opt in ("-j", "height="):
            height = int(arg)
        elif opt in ("-p", "padding="):
            padding = int(arg)
    todoList = yaml.load(open(configFile))
    champs = todoList["champions"]
    creeps = todoList["creatures"]
    champProbs = todoList["champion_probabilities"]
    creepProbs = todoList["creature_probabilities"]
    creepScales = todoList["creature_scale_factors"]
    for i, creep in enumerate(creeps):
        creepScaleFactors[creep] = creepScales[i]
    countFile = inputFile + inFilePrefix + "/" + countFile
    countDic = yaml.load(open(countFile))
    #outputFile = outputFile + inFilePrefix + "/"
    if DEBUG:
        print("Working in Debug Mode!")
    print("The following champions and creatures will be in the images:")
    for c in countDic.keys():
        print("--", c)
    print("The data set can be found in:", outputFile)
    print("Input data will be from:", outputFile + "images")
    print("Images will be (", width, ",",height, ")")
    print("Images with bounding boxes will be in:", outputFile + "keys")
    print("Mask images will be from: ", inputFile +  inFilePrefix)
    print("Map images will be from:", mapFile)
    print("Will  make a data set consisting of", numImages, "raw images, bounding box images, and label files.")
    return (champs, creeps, countDic, champProbs, creepProbs)

def writeLabelFile(champs, creeps):
    global labelFile, labels, outputFile
    if not os.path.exists(outputFile):
        os.makedirs(outputFile)
    idx = 0
    with open(outputFile +  "labels.txt", "w") as f:
        for champ in champs:
            f.write(champ + "\n")
            labels[champ] = idx
            idx = idx + 1
        for creep in creeps:
            f.write(creep + "\n")
            labels[creep] = idx
            idx = idx + 1
            
def choices(dist, weight):
    p = random()
    tot = 0.0
    for i in range(len(dist)):
        tot = tot + weight[i]
        if p < tot:
            return dist[i]
    print("Error in choices")
        
def getChamps(champs, countDic, dist):
    num = choices(dist['numChamps'][0],dist['numChamps'][1])
    inChamps = [choices(champs,dist['champProbs']) for i in range(num)]
    imgs = []
    masks = []
    names = []
    for i in range(len(inChamps)):
        idx = str(randint(1,countDic[ inChamps[i]]))
        imgs.append(Image.open(inputFile + inFilePrefix + "/" + inChamps[i] +"/" +  inChamps[i] + idx + ".jpg"))
        masks.append(Image.open(inputFile + inFilePrefix + "/" +  inChamps[i] +"/" +  inChamps[i] + idx + "_mask.jpg").convert("L"))
    return (imgs, masks, inChamps)

def getCreeps(creeps, countDic, dist):
    num = choices(dist['numCreeps'][0], dist['numCreeps'][1])
    inCreeps = [choices(creeps,dist['creepProbs']) for i in range(num)]
    imgs = []
    masks = []
    names = []
    for creep in inCreeps:
        idx = str(randint(1,countDic[creep]))
        imgs.append(Image.open(inputFile + inFilePrefix + "/" + creep +"/" + creep + idx + ".jpg"))
        masks.append(Image.open(inputFile + inFilePrefix + "/" + creep +"/" + creep +idx + "_mask.jpg").convert("L"))
        names.append(creep)
    return (imgs, masks, names)
                               
                               
def rot(champs, champMasks, creeps, creepMasks):
    global maxRot
    champRots = []
    champMaskRots = []
    creepRots = []
    creepMaskRots = []
    for i in range(len(champs)):
        rotAngle = np.random.normal(loc=0.0, scale = maxRot)
        champRots.append(champs[i].rotate( rotAngle, expand=True))
        champMaskRots.append(champMasks[i].rotate(rotAngle, expand = True))
    for i in range(len(creeps)):
        rotAngle = np.random.normal(loc=0.0, scale = maxRot)
        creepRots.append(creeps[i].rotate(rotAngle, expand=True))
        creepMaskRots.append(creepMasks[i].rotate(rotAngle, expand=True))

    return (champRots, champMaskRots, creepRots, creepMaskRots)

def resize(champs, champMasks, creeps, creepMasks,creepList):
    global maxScale, minScale, creepScaleFactors
    scale = uniform(minScale, maxScale)
    sChamp = []
    sChampMask = []
    sCreep = []
    sCreepMask = []
    for i in range(len(champs)):
        w,h = champs[i].size
        w = int(scale * w)
        h = int(scale * h)
        sChamp.append(champs[i].resize((w,h)))
        sChampMask.append(champMasks[i].resize((w,h)))
    for i in range(len(creeps)):
        w,h = creeps[i].size
        w = int(scale * w * creepScaleFactors[creepList[i]])
        h = int(scale * h * creepScaleFactors[creepList[i]])
        sCreep.append(creeps[i].resize((w,h)))
        sCreepMask.append(creepMasks[i].resize((w,h)))
        
    return (sChamp, sChampMask, sCreep, sCreepMask)

def getMapImg():
    global mapImgs
    return choice(mapImgs).copy()
    
def addNoise(frame):
    pass

def place(mapImg, champs, champMasks, creeps, creepMasks, champList, creepList):
    # to-do add the HUD
    global labels
    wMap, hMap = mapImg.size
    labelData = []
    for i in range(len(champs)):
        w,h = champs[i].size
        uv =  (randint(0,wMap-w),randint(0,hMap-h))
        mapImg.paste(champs[i], uv, champMasks[i])
        # class, cx, cy, wid, height (raw)
        labelData.append([ labels[champList[i]], uv[0]+w/2, uv[1]+h/2, w, h])
    (cx, cy) = getCreepLoc(creeps[0], wMap, hMap)
    w,h = creeps[0].size
    mapImg.paste(creeps[0], (cx,cy), creepMasks[0])
    labelData.append([labels[creepList[0]], cx+w//2, cy+h//2, w, h])
    for i in range(1, len(creeps)):
        (x,y) = getCreepLoc(creeps[i], wMap, hMap, cx, cy)
        mapImg.paste(creeps[i], (x,y), creepMasks[i])
        if y > 1080:
            print (x,y)
        labelData.append([ labels[creepList[i]], x+w//2, y+h//2, w, h])

    return mapImg, labelData


def getCreepLoc(creep, wMap, hMap, cx=None, cy=None):
    global creepVar
    w,h = creep.size
    if cx == None and cy == None:
        cx = randint(w//2,wMap-w//2)
        cy = randint(h//2,hMap-h//2)
        return (cx, cy)
    else:
        safe = False
        while not safe:
            x = int(np.random.normal(loc = cx,scale=creepVar))
            y = int(np.random.normal(loc=cy,scale=creepVar))
            if x > 0 and x < wMap-w-1 and y>0 and y < hMap-h-1:
                safe = True
        return (x,y) 
        
        
def drawRect(frame, x, y, w, h, labe):
    global labelColor
    frame1 = ImageDraw.Draw(frame)
    frame1.line([(x-w/2,y-h/2),(x+w/2,y-h/2),(x+w/2,y+h/2),(x-w/2,y+h/2),(x-w/2,y-h/2)],
                          labelColor[labe], width = 3)
    return frame
def pad(frame):
    global padding, width, height
    result = Image.new(frame.mode, (width, height), (0,0,0))
    result.paste(frame, (int(padding/2), 0))
    return result
        
def save(frame, labelData, i):
    global outputFile, width, height, padding
    if not os.path.exists(outputFile  + "images/"):
        os.makedirs(outputFile + "images/")
    if not os.path.exists(outputFile + "labels/"):
        os.makedirs(outputFile + "labels/")
    if not os.path.exists(outputFile + "key/"):
        os.makedirs(outputFile + "key/")
    initWidth, initHeight = frame.size
    key = frame.copy()
    frame = frame.resize((width-padding, height))
    frame = pad(frame)
    fil = open(outputFile + "labels/label" + str(i) + ".txt", "w+")
    for data in  labelData:
        key = drawRect(key, data[1], data[2], data[3], data[4], data[0])
        outString = str(data[0]) + " " + str(data[1]/initWidth) + " " +  str(data[2]/initHeight) +" " + str(data[3]/initWidth) + " " +  str(data[4]/initHeight) + "\n"
        fil.write(outString)
        if (data[1]/initWidth < 0.0 or data[2]/initHeight > 1.0 or data[3]/initWidth<0.0 or data[4]/initHeight > 1.0):
            print("Init:", (initWidth, initHeight))
            print("Data Values:", data[1], data[2], data[3], data[4])
            print("File: " + outString)
            print("Index:", i)
            print()
            
    key = key.resize((width-padding, height))
    key = pad(key)
    frame.save(outputFile + "images/image" + str(i) + ".jpg")
    key.save(outputFile + "key/key" + str(i) + ".jpg")
    
def show(frame, name):
    cv2.imshow(name, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
def show2(frame, name):
    cv2.imshow(name, cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA))
    
if __name__ == "__main__":
    print("\nStarting bootstrap")
    (champs, creeps, countDic, champProbs, creepProbs) = unpack(sys.argv[1:])
    mapImgs = [Image.open(mapFile + f) for f in os.listdir(mapFile)]
    dist = {}

    # champ selection info
    numChamps = range(maxChamp+1)
    champDist = [1.0 - haveChamp]
    for i in range(maxChamp):
        champDist.append(haveChamp/maxChamp)
    dist['numChamps'] = [numChamps, champDist]

    # champ dist
    

    # creep selection info
    numCreeps = range(maxCreep+1)
    hCreepDist = [1.0 - haveCreep]
    for i in range(maxCreep):
        hCreepDist.append(haveCreep/maxCreep)
    dist['numCreeps'] = [numCreeps, hCreepDist, creepDist]

    dist['champProbs'] = champProbs
    dist['creepProbs'] = creepProbs
    writeLabelFile(champs, creeps)

    for i in range(numImages):
        # get the champs/creeps masks to use
        (champImgs, champMasks, champList) = getChamps(champs, countDic, dist)
        (creepImgs, creepMasks, creepList) = getCreeps(creeps, countDic, dist)
        
        # randomly rotate + add noise to each mask
        (champImgs, champMasks, creepImgs, creepMasks) = rot(champImgs, champMasks, creepImgs, creepMasks)
        (champImgs, champMasks, creepImgs, creepMasks) = resize(champImgs, champMasks, creepImgs, creepMasks, creepList)
        # place the masks in the image
        mapImg = getMapImg()
        frame, labelData = place(mapImg, champImgs, champMasks, creepImgs, creepMasks, champList, creepList)
        if DEBUG:
            show2(frame, "Map Image")
            cv2.waitKey(0)
        #frame = addNoise(frame)
        # generate key and save
        save(frame, labelData, i)
