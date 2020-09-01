from numpy import load
import numpy as np
from cv2 import *
from math import fabs
import sys

fileName = "../DeepLeagueData/clusters_cleaned/test/data_test_set_cluster_0.npz"
classPath = "../DeepLeagueData/league_classes.txt"
picPath = "../champion/"

data = load(fileName, allow_pickle=True)

items = data.files
imgs = data['images']
boxes = data['boxes']

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# 2,3,4  sucks
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
        print picPath + classNames[idx] + ".png"
        data.append( ( cv2.resize(cv2.imread(picPath + classNames[idx] + ".png"), (24,24)),
                     classNames[idx]))
    return data
                    
                
if __name__ == "__main__":
    classNames = get_classes(classPath)
    method = methods[1]
    print "Found using Method:", method
    for i in range(4):
        img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
        print img.shape
        img2 = img.copy()
        classIdxs = []
        for b in boxes[i]:
            classIdxs.append(b[0])
        classData = getClassPics(classIdxs, classNames)

        for temp, name in classData:
            img = img2.copy()
            cv2.imshow("Template for " + name, temp)
            w = temp.shape[0]
            h = temp.shape[1]
            res = cv2.matchTemplate(img,temp,eval(method))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            cv2.rectangle(img,top_left, bottom_right, (0,0,255), 2)
            cv2.imshow("Matching Result for " + name, res)
            cv2.imshow("Detected Points", img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    cv2.destroyAllWindows()
    sys.exit()
