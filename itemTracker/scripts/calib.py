


import cv2
import numpy as np
import os, sys
import yaml

RAWINPUT = False

# default locations (for testing)
imgPath = "../data/6items_1player.png"
cfgOutput = "../cfg/itemPts.yaml"

if __name__=="__main__":
    ptsDict = {}
    if RAWINPUT:
        imgPath = raw_input("Enter relative file path for image:")
    ogImg = cv2.imread(imgPath)
    if RAWINPUT:
        u = int(raw_input("Enter initial u guess:\t"))
        v = int(raw_input("Enter initial v guess:\t"))
    else:
        u = 400
        v = 300
    print "Initial guess: (", u, ",", v, ")"
    while True:
        img = ogImg.copy()
        
        cirImg = cv2.circle(img, (u,v), 7, (0, 0, 255))
        cirImg = cv2.circle(img, (u,v), 1, (0, 0, 255))
        cv2.imshow("Calibration Image", cirImg)
        key = chr(cv2.waitKey(0))
        
        if key == "w":
            v = v - 1
        elif key=="s":
            v = v+1
        elif key=="a":
            u = u-1
        elif key=="d":
            u = u + 1
        elif key=="q":
            print "Final:\t (", u, ",", v, ")"
            break
        elif key=="e":
            name = raw_input("Enter Var Name:\t")
            print "Final:\t (", u, ",", v, ")"
            pts[name] = (u,v)
            
    if RAWINPUT:
        cfgOutput = raw_input("Enter File address:\t")
                              
    with open(cfgOutput,'w') as file:
        yaml.dump(ptsDict, file)
    cv2.destroyAllWindows()
    sys.exit()


    
    
