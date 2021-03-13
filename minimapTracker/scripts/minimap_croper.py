from numpy import load
import numpy as np
from cv2 import *
import sys

npzPath = "../DeepLeagueData/clusters_cleaned/"
classPath = "../DeepLeagueData/league_classes.txt"
portPath = "../champion/"
outpath = "../minimapPortraits/"


# stolen from DeepLeague (farzaa)
def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def getClassPics(classIdxs, classNames):
    global picPath
    data = []
    for idx in classIdxs:
        print( picPath + classNames[idx] + ".png")
        data.append( ( cv2.resize(cv2.imread(picPath + classNames[idx] + ".png"), (24,24)),
                     classNames[idx]))
    return data

if __name__ == "__main__":

    #load data
    classNames = get_classes(classPath)
    failed = False
    for phase in ['test','val','train']:
        imgCount = 0
        oldCount = 0
        print(phase)
        for k in range(8):
            print("\tData Set #" + str(k))
            try:
                data = load(npzPath + phase +
                        "/data_" + phase + "_set_cluster_" + str(k) + ".npz",
                        allow_pickle=True)
                failed = False
            except:
                print("\t\tFailed to open")
                failed = True
                continue
                
            imgs = data['images']
            boxes = data['boxes']

            for i in range(len(imgs)):
                # load minimap image
                img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
                # cropout each champion portrait
                cropImgs = []
                champIdx = []
                for b in boxes[i]:
                    (label, minX, minY, maxX, maxY) = b
                    cropImgs.append(img[minY+1:maxY-1, minX+1:maxX-1,:]) # make it 28x28
                    champIdx.append(label)

                # save crop images as img(imgCount)_(label).png
                for j in range(len(cropImgs)):
                    cv2.imwrite(outpath + "/" + phase +
                                "/img" + str(imgCount) + "_" + str(label) + ".png",
                                cropImgs[j])
                    imgCount = imgCount + 1
            
            if not failed:
                print("\t\tCreated " + str(imgCount-oldCount) + " portrait images")
            oldCount = imgCount
        
        print("\t" + phase + " folder has a total of " + str(imgCount) + " images")
        
    
