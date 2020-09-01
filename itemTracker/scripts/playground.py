

import cv2
import numpy as np
import os, sys
from random import choice
import time
itemFile = "../data/item/"

# load all item images
files = os.listdir(itemFile)
hists = 4
# With color histograms

def getHist(img):
    hist = cv2.calcHist([img], [0,1,2], None, [hists, hists, hists], [0,256,0,256,0,256])
    hist = cv2.normalize(hist,hist, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    print hist
    return hist

def getMatch(testHist):
    eps = 1e-5
    global itemHists
    for i, itemHist in enumerate(itemHists):
        if cv2.compareHist(testHist, itemHist, cv2.HISTCMP_CHISQR) == 0.0:
            return i

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
                    
                        
items = [cv2.imread(itemFile + x) for x in files]
itemHists = [getHist(item) for item in items]
if __name__ == "__main__":
    tt = 0.0
    iters = 20
    for i in range(iters):
        # select random image to match
        test = choice(items)
        strt = time.time() # here is what we would do in real time
        testHist = getHist(test)
        i = getMatch(testHist)
        dT = time.time() - strt
        tt = tt + dT
        #print "The resulting simularity:", results[i]
        
        print "It took", dT, "seconds to find a match"
        cv2.imshow("Looked for pic", test)
        cv2.imshow("Found Pic", items[i])
        cv2.waitKey(1000)
    print "Average Time was:", tt/iters
    sys.exit()


