import cv2
import numpy as np
import yaml
import sys, os
from pynput import keyboard
from random import choice

"""

Takes imgPath file and extracts image matches

to-do:   corrolate images to league data
         Extract numbers from images
         Red Side Extraction
         place into class formate


"""
#cfgFile = "../cfg/itemPts.yaml"
cfgFile = "../cfg/testItem.yaml"
cfgOutFile = "../cfg/itemOffsets.yaml"
imgPath = "../data/screen/items17.png"
outFile = "../results/item_extraction/"
itemFile = "../data/item/"

# controls behavior
WRITEIMGS = False
UPDATE_CFG = False
RAW = False
VISUALIZE = True
NEWIMG = False
RUNNING = True
# d indicates it is an offset
bBoxDx = None
bBoxDy = None
firstItemD =  None
nextItemD = None
offset = None
topLeft = None
HISTS = 4


itemHists = []
def getHist(img):
    global HISTS
    hist = cv2.calcHist([img], [0,1,2], None, [HISTS, HISTS, HISTS], [0,256,0,256,0,256])
    hist = hist/255.0
    #hist = cv2.normalize(HISTS,HISTS, 0, 255 , cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    return hist

def getMatch(testHist):
    eps = 1e-5
    global itemHists
    score = 1000000.0
    idx = 0
    for i, itemHist in enumerate(itemHists):
        
        newScore =  cv2.compareHist(testHist, itemHist, cv2.HISTCMP_CHISQR)
        if newScore < score:
            idx = i
            score = newScore
    return idx

def getMatch2(testHist, eps = 1e-10):
    global itemHists
    con = False
    for idx, itemHist in enumerate(itemHists):
        for i in range(4):
            if con == True:
                continue
            for j in range(4):
                if con==True:
                    continue
                for u in range(4):
                    if abs(testHist[i][j][u] - itemHist[i][j][u]) > eps:
                        con = True
                        continue
                    elif i == 3 and j == 3 and u == 3:
                        return idx
        if con==True:
            con = False
            continue
                    
                        
def subTup(a, b):
    return np.array( [a[0] - b[0], a[1]-b[1]] )

def crop(img, a, b):
    return img[a[1]:b[1], a[0]:b[0]]


def extBigBoxes(img, bBoxDx, bBoxDy, topLeft):
    """ Removes 5 blocks of big images and retuns them in array """
    top = topLeft.copy()
    top[1] = top[1] - 3
    bot = topLeft.copy()
    bot[0] = bot[0] + bBoxDx
    
    bBoxes = []
    for i in range(5):
        #top[1] = top[1] + bBoxDy
        top[1] = bot[1]
        bot[1] = bot[1] + bBoxDy
        bBoxes.append(crop(img.copy(), top, bot))

    return bBoxes

def extItemImg(bBox, firstItemD, nextItemD, offset = 0):
    """ Removes the images for 7 items (6 items + trinket) """

    top = firstItemD.copy()
    bot = top.copy() + nextItemD
    bot[0] = bot[0] + offset
    items = []
    for i in range(7):
        items.append(crop(bBox.copy(), top, bot))
        top[0] = top[0] + nextItemD[0] + offset
        bot[0] = bot[0] + nextItemD[0] + offset

    return items

def readFromRawConfig(ptsDict):
    
    global firstItemD, nextItemD, offset, bBoxDx, bBoxDy, topLeft
    bBoxDx = ptsDict["blue_bot_right"][0] - ptsDict["blue_top_left"][0]
    bBoxDy = ptsDict["blue_bot_right"][1] - ptsDict["blue_top_left"][1] - 5
    firstItemD = subTup(ptsDict["blue_left_item_top_left"], ptsDict["blue_top_left"])
    nextItemD = subTup(ptsDict["blue_left_item_bot_right"], ptsDict["blue_left_item_top_left"])
    topLeft = np.array(ptsDict["blue_top_left"])

def readFromProcessedConfig(calData):
    global firstItemD, nextItemD, offset, bBoxDx, bBoxDy, topLeft
    bBoxDx = calData["BigBoxDX"]
    bBoxDy = calData["BigBoxDY"]
    topLeft = np.array(calData["TopLeft"])
    firstItemD = np.array(calData["FirstItemOffset"])
    nextItemD = np.array(calData["NextItem"])
    offset = calData["ItemOffset"]
    
def writeProcessedConfig(filename):
    global firstItemD, nextItemD, offset, bBoxDx, bBoxDy, topLeft
    calData = {}
    calData["BigBoxDX"] = bBoxDx
    calData["BigBoxDY"] = bBoxDy
    calData["TopLeft"] = topLeft.tolist()
    calData["FirstItemOffset"] = firstItemD.tolist()
    calData["NextItem"] = nextItemD.tolist()
    calData["ItemOffset"] = 1
    with open(cfgOutFile,'w') as file:
        yaml.dump(calData, file)

def onTab(key):
    global NEWIMG, RUNNING
    if key == keyboard.Key.tab:
        NEWIMG = True
    elif key == keyboard.Key.esc:
        RUNNING = False
        return False

# note top left is (0,0)
if __name__ == "__main__":
    
    # load all item images/hists
    files = os.listdir(itemFile)
    itemImgs = [cv2.resize(cv2.imread(itemFile + x), (34, 32)) for x in files]
    itemHists = [getHist(item) for item in itemImgs]
    
    img = cv2.imread(imgPath)
    if RAW:
        ptsDict = yaml.load(open(cfgFile), Loader = yaml.FullLoader)
        readFromRawConfig(ptsDict)
    else:
        calDict = yaml.load(open(cfgFile), Loader = yaml.FullLoader)
        readFromProcessedConfig(calDict)

    """
    with keyboard.Listener(
            on_press=onTab) as listener:
        listener.join()
    """  
    listener = keyboard.Listener(on_press=onTab)
    listener.start()
    
    while True:
        if NEWIMG:
            bBoxes = extBigBoxes(img, bBoxDx, bBoxDy, topLeft)
    
            for i, bb in enumerate(bBoxes):
                items = extItemImg(bb, firstItemD, nextItemD, offset=1)
                if WRITEIMGS:
                    cv2.imwrite(outFile + str(i) + "_champ_box.png", bb)
                if VISUALIZE:
                    cv2.imshow("Big Box" + str(i), bb)
                itemIter = iter(items)
                j = 0
                for item in itemIter:
                    testHist = getHist(item)
                    idx = getMatch(testHist)
                    if idx == 199:
                        for k in range(6 - j - 1):
                            next(itemIter)
                        j = 6
                        continue
                    if VISUALIZE:
                        cv2.imshow("Item " + str(j), item)
                        cv2.imshow("Item " + str(j) + " match", itemImgs[idx])
                    if WRITEIMGS:
                        cv2.imwrite(outFile + str(i) + "_" + str(j) + "_box.png", item)
                        cv2.imwrite(outFile + str(i) + "_" + str(j) + "_match.png", itemImgs[idx])
                    j = j + 1
                cv2.waitKey(0)
                #cv2.destroyAllWindows() 
            if VISUALIZE:
                cv2.waitKey()
            NEWIMG = False
        elif not RUNNING:
                break
        
    if VISUALIZE:
        cv2.destroyAllWindows()
    if UPDATE_CFG:
        writeProcessedConfig(cfgOutFile)
    sys.exit()
