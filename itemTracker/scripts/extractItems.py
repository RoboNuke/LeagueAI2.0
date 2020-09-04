import cv2
import numpy as np
import yaml
import sys
from pynput import keyboard

cfgFile = "../cfg/itemPts.yaml"
cfgOutFile = "../cfg/itemOffsets.yaml"
imgPath = "../data/screen/items17.png"
outFile = "../results/item_extraction/"

# controls behavior
WRITEIMGS = False
UPDATE_CFG = False
RAW = True
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
    cv2.imshow("Testy", img)
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
        print("New Img" + str(NEWIMG))
    elif key == keyboard.Key.esc:
        RUNNING = False
        return False

    
# note top left is (0,0)
if __name__ == "__main__":
    img = cv2.imread(imgPath)
    if RAW:
        #ptsDict = yaml.load(open(cfgFile), Loader = yaml.FullLoader)
        ptsDict = yaml.load(open(cfgFile))
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
            print("fOUND NEW IMAGE")
            bBoxes = extBigBoxes(img, bBoxDx, bBoxDy, topLeft)
    
            for i, bb in enumerate(bBoxes):
                items = extItemImg(bb, firstItemD, nextItemD, offset=1)
                cv2.imwrite(outFile + str(i) + "_champ_box.png", bb)
                if VISUALIZE:
                    cv2.imshow("Big Box" + str(i), bb)
                for j, item in enumerate(items):
                    if VISUALIZE:
                        cv2.imshow("Item " + str(j), item)
                    if WRITEIMGS:
                        cv2.imwrite(outFile + str(i) + "_" + str(j) + "_box.png", item)
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
