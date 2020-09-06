from time import sleep
from pynput import *

import cv2
import numpy as np
import pyautogui

upsydown = 676
dright = 730

udPer = 40
rPer = 60

scrnStrt = (0,0)
screenSize = (1920, 1080)

imgCount = 0
outFile = "../data/0map/"

sTime = 0.001
def saveScreen():
    global imgCount
    img = pyautogui.screenshot(region = (scrnStrt[0], scrnStrt[1], screenSize[0], screenSize[1]))
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outFile + "map" + str(imgCount) + ".jpg", frame)
    imgCount = imgCount + 1
    

def moveUp(board):
    board.tap(keyboard.Key.up)
    sleep(sTime)

def moveDown(board):
    board.tap(keyboard.Key.down)
    sleep(sTime)

def moveRight(board):
    board.tap(keyboard.Key.right)
    sleep(sTime)

def gotoTopBot(k, top=True):
    global upsydown, udPer
    for i in range(int(upsydown/udPer)+1):
        saveScreen()
        for j in range(udPer):
            if top:
                moveUp(k)
            else:
                moveDown(k)

if __name__=="__main__":
    # Assuming the window is already open
    sleep(5)
    print("Starting Process")
    k = keyboard.Controller()
    topy = True
    print(topy)
    moveRight(k)
    for i in range(int(dright/rPer)+1):
        gotoTopBot(k, top=topy)
        topy = not topy
        print(topy)
        for j in range(rPer):
            moveRight(k)
        #sleep(0.5)
    print("Completed")
