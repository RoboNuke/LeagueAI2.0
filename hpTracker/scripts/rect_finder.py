import cv2
import imutils
import numpy as np
import time
import os, sys
def detect(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) == 4:
        (x,y,w,h) = cv2.boundingRect(approx)
        return("Rectangle")
    else:
        return(-1)


def getLines(frame):
    thresh = threshImage(frame.copy())
    #cv2.imshow("Red Only", thresh)
    
    #thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grey", thresh)
    edges = cv2.Canny(thresh,50,150,apertureSize = 3)
    cv2.imshow("Edges", edges)
    """
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
    """
    
    minLineLength = 15
    maxLineGap = 3
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),4)
    return frame

def getRects(frame):
    # preprocess img
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Grey Image", img)
    #img = cv2.GaussianBlur(frame, (15,15), 0)
    thresh = threshImage(frame.copy())
    cv2.imshow("Red Only", thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        shape = detect(c)
        if shape != -1:
            #M = cv2.moments(c)
            #cX = int(M["m10"] / M["m00"])
            #cY = int(M["m01"] / M["m00"])

            cv2.drawContours(frame, [c], -1, (0,255, 0), 2)
            #cv2.putText( frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

    return(frame)

def threshImage(frame):
    (b,g,r) = cv2.split(frame)
    bn = cv2.inRange(b, 0, 15)
    gn = cv2.inRange(g, 16, 255)
    rn = cv2.inRange(r, 56, 255)

    thresh2 = bn - gn - rn
    return(thresh2)


    h = frame.shape[0]
    w = frame.shape[1]
    out = np.zeros([h,w], dtype=np.uint8)
    for u in range(h):
        for v in range(w):
            pt = frame[u][v]
            #if( pt[0] > 40 and pt[0] < 95 and  #blue
            #    pt[1] > 40 and pt[1] < 85 and #green
            #    pt[2] > 85 and pt[2] < 161): # red
            if( pt[0] < 15 and pt[1] < 15 and pt[2] < 55 ):
                out[u][v] = 255               
    return(out)

def convolv(frame, mat):
    h = frame.shape[0]
    w = frame.shape[1]

    mW = mat.shape[1]
    mH = mat.shape[0]

    conv = np.zeros([h - mH + 1,w - mW + 1], dtype=np.uint8)
    for u in range(h - mH + 1):
        for v in range(w - mW + 1):
            roi = frame[u:u+mH, v:v+mW]
            conv[u,v] = (roi * mat).sum()
    return conv
    
if __name__ == "__main__":
    # create kernal
    matValue = 1.0/130.0
    mat = np.zeros([6, 61])
    mat[0,:] = matValue
    mat[-1,:] = matValue
    mat[:,0] = matValue
    mat[:,-1] = matValue

    # load all images
    testImgFile = "../data/images/"
    files = os.listdir(testImgFile)
    print("Will look at {} images".format(len(files)))
    for imgName in files:
        
        img = cv2.imread(testImgFile + imgName)
        cv2.imshow("Original", img)
        start = time.time()
        img[0:130,0:370] = np.ones([130,370,3]) * 255
        img = img[0:800,:]
        
        frame = threshImage(img.copy())
        threshTime = time.time()
        #print("Time to threshold image:", threshTime - start)
        start = time.time()
        #frame = convolv(frame, mat)  # my funct takes 5 sec
        frame = cv2.filter2D(frame, -1, mat) # opencv 0.036
        #cv2.imshow("Convolved Image", frame)
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(frame)
        healthBars = np.argwhere( frame > 220 )
        convTime = time.time()
        #print("Convolution Time:", convTime - start)
        for loc in healthBars:
            cv2.rectangle(img, (loc[1] - 30, loc[0] - 3),  (loc[1] + 31, loc[0] + 3), (255,0,0),2)
        
        cv2.imshow("Final", img)
        cv2.waitKey(0)
