import cv2
import numpy as np
import os

champPath = "../champion/"
outPath = "../preped_champ/"

size = 28
r = 26

mask = np.zeros([28,28,3],dtype=np.uint8)

mask = cv2.circle(mask, (14,14), 10, (1,1,1), -1)

for imgName in os.listdir(champPath):
    champImg = cv2.resize(cv2.imread(champPath + imgName), (24,24))
    champImg = cv2.copyMakeBorder( champImg, 2, 2, 2, 2, cv2.BORDER_CONSTANT,value=(0,0,0))
    comp = cv2.multiply(champImg,mask)
    cv2.imwrite(outPath + imgName, comp)
