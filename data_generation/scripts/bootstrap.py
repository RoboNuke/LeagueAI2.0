

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

#Globals

configFile = "../cfg/vayne.yaml"
countFile = "count.yaml"
outputFile = "../data/3data_set/"
inputFile = "../data/2cropped_images/"
inFilePrefix = "test"
mapFile = "../data/0map/"
labelFiles = "../cfg/vayne.names"
numImages = 10
labels = {"Summoners Rift Chaos Minion Melee":0,
          "Vayne":1}
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
creepVar = 75 # pixels

# Scaling 
minScale = .35
maxScale = .5


mapImgs = []

def unpack(argv):
    global outputFile, inputFile, countFile, inFilePrefix, DEBUG, mapFile, numImages, configFile
    try:
        opts, args = getopt.getopt(argv,"hc:o:i:k:q:m:n:l:d", ["configFile=", "outputFile=","inputFile=", "countFile=","inFilePrefix=","mapFile=", "numberOfImages=","labelFile=", "debug="])
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
            numImages = arg
        elif opt in ("-l", "labelFile="):
            labelFile = arg
    todoList = yaml.load(open(configFile))
    champs = todoList["champions"]
    creeps = todoList["creatures"]
    
    countFile = inputFile + inFilePrefix + "/" + countFile
    countDic = yaml.load(open(countFile))
    
    if DEBUG:
        print("Working in Debug Mode!")
    print("The following champions and creatures will be in the images:")
    for c in countDic.keys():
        print("--", c)
    print("The data set can be found in:", outputFile)
    print("Input data will be in:", outputFile + "images")
    print("Key infromation will be in:", outputFile + "keys")
    print("Mask images will be from: ", inputFile + "/" + inFilePrefix)
    print("Map images will be from:", mapFile)
    print("Will  make", numImages, "raw data images.")
    return (champs, creeps, countDic)

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
    inChamps = [choice(champs) for i in range(num)]
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
    imgs = []
    masks = []
    names = []
    for i in range(num):
        #print creeps
        #print dist['numCreeps'][2]
        creep = choices(creeps, dist['numCreeps'][2])
        idx =  str(randint(1,countDic[creep]))
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

def resize(champs, champMasks, creeps, creepMasks):
    global maxScale, minScale
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
        w = int(scale * w)
        h = int(scale * h)
        sCreep.append(creeps[i].resize((w,h)))
        sCreepMask.append(creepMasks[i].resize((w,h)))
        
    return (sChamp, sChampMask, sCreep, sCreepMask)

def getMapImg():
    global mapImgs
    return choice(mapImgs)
    
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
    cx = randint(creepB, wMap - creepB)
    cy = randint(creepB, hMap - creepB)
    uvs = [(int(np.random.normal(loc = cx,scale=creepVar)), int(np.random.normal(loc=cy,scale=creepVar))) for i in range(len(creeps)) ]
    for i in range(len(creeps)):
        w,h =  creeps[i].size
        mapImg.paste(creeps[i], uvs[i], creepMasks[i])
        labelData.append([ labels[creepList[i]], uvs[i][0]+w/2, uvs[i][1]+h/2, w, h])

    return mapImg, labelData

labelColor={ 0: ImageColor.getrgb("blue"),
            1: ImageColor.getrgb("red") }
def drawRect(frame, x, y, w, h, labe):
    global labelColor
    frame1 = ImageDraw.Draw(frame)
    frame1.line([(x-w/2,y-h/2),(x+w/2,y-h/2),(x+w/2,y+h/2),(x-w/2,y+h/2),(x-w/2,y-h/2)],
                          labelColor[labe], width = 3)
    return frame
def save(frame, labelData, i):
    global outputFile
    if not os.path.exists(outputFile + "images/"):
        os.makedirs(outputFile + "images/")
    if not os.path.exists(outputFile + "labels/"):
        os.makedirs(outputFile + "labels/")
    if not os.path.exists(outputFile + "key/"):
        os.makedirs(outputFile + "key/")
    w,h = frame.size
    frame.save(outputFile + "images/image" + str(i) + ".jpg")
    fil = open(outputFile + "labels/label" + str(i) + ".txt", "w+")
    for data in  labelData:
        frame = drawRect(frame, data[1], data[2], data[3], data[4], data[0])
        fil.write(str(data[0]) + " " + str(data[1]) + " " +  str(data[2]) + " " + str(data[3]) + " " +  str(data[4]) + "\n")
    frame.save(outputFile + "key/key" + str(i) + ".jpg")
    
def show(frame, name):
    cv2.imshow(name, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
def show2(frame, name):
    cv2.imshow(name, cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA))
    
if __name__ == "__main__":
    (champs, creeps, countDic) = unpack(sys.argv)
    mapImgs = [Image.open(mapFile + f) for f in os.listdir(mapFile)]
    dist = {}

    # champ selection info
    numChamps = range(maxChamp+1)
    champDist = [1.0 - haveChamp]
    for i in range(maxChamp):
        champDist.append(haveChamp/maxChamp)
    dist['numChamps'] = [numChamps, champDist]

    # creep selection info
    numCreeps = range(maxCreep+1)
    hCreepDist = [1.0 - haveCreep]
    for i in range(maxCreep):
        hCreepDist.append(haveCreep/maxCreep)
    dist['numCreeps'] = [numCreeps, hCreepDist, creepDist]

    
    for i in range(numImages):
        # get the champs/creeps masks to use
        (champImgs, champMasks, champList) = getChamps(champs, countDic, dist)
        (creepImgs, creepMasks, creepList) = getCreeps(creeps, countDic, dist)
        
        # randomly rotate + add noise to each mask
        (champImgs, champMasks, creepImgs, creepMasks) = rot(champImgs, champMasks, creepImgs, creepMasks)
        (champImgs, champMasks, creepImgs, creepMasks) = resize(champImgs, champMasks, creepImgs, creepMasks)
        # place the masks in the image
        mapImg = getMapImg()
        frame, labelData = place(mapImg, champImgs, champMasks, creepImgs, creepMasks, champList, creepList)
        show2(frame, "Map Image")
        cv2.waitKey(0)
        #frame = addNoise(frame)
        # generate key and save
        save(frame, labelData, i)
        print("Next Image")
