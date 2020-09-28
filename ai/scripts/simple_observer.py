from ctypes import *
import random
import os
import cv2
import time
import sys
DARKNETPATH = os.environ.get('DARKNET_PATH')
sys.path.insert(1, DARKNETPATH)
import darknet
import argparse
from threading import Thread, enumerate
#from queue import Queue
import numpy as np
import pyautogui
import mss
import multiprocessing 
from multiprocessing import Queue


def parser():
    parser = argparse.ArgumentParser(description="Simple Vayne State Observer")

    parser.add_argument("--cfg", type=str,
                        default=DARKNETPATH + "cfg/yolov3-vayne.cfg",
                        help="path to network configuration file")

    parser.add_argument("--weights", type=str,
                        default=DARKNETPATH + "backup/vayne/yolov3-vayne_last.weights",
                        help="path to network pre-trained weight file")

    parser.add_argument("--display", type=bool, default=True,
                        help="Set to true to create a second window displaying the state")

    parser.add_argument("--data_file", type=str,
                        default=DARKNETPATH + "data/vayneDataset/vayne.data",
                        help = "path to data file")
    
    parser.add_argument("--thresh", type=float, default=0.25,
                        help="remove detections with confidence below this value")

    parser.add_argument("--padding", type=int, default=4,
                        help = "Padding to add half of above and below the image")
    return parser.parse_args()

class Observation():
    def __init__(self, classIdx, prob, x, y):
        self.idx = classIdx
        self.prob = prob
        self.x = x
        self.y = y
        

class Observer(object):
    def __init__(self, args):
        print("Initializing Observer")
        self.args = args
        self.RUNNING = True
        self.padding = args.padding
        
        # set up network
        self.network, self.classNames, self.classColors = darknet.load_network(
            self.args.cfg,
            self.args.data_file,
            self.args.weights,
            batch_size=1
        )

        # set up queues
        #self.dImgQue = Queue(maxsize=1)
        self.msgQue = Queue(maxsize=1)
        self.imgQue = Queue()
        self.fpsQue = Queue(maxsize=1)
        self.detQue  = Queue(maxsize=1)
        self.stateQue = Queue(maxsize=1)

        self.initWidth = 1920
        self.initHeight = 1080
        
        # create darknet img
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.dImg = darknet.make_image(self.width, self.height, 3)

        
    def start(self):
        multiprocessing.Process(target=self.imgCapture, args=(self.imgQue, self.msgQue,)).start()
        multiprocessing.Process(target=self.drawing,
                                args=(self.imgQue, self.detQue, self.fpsQue, self.msgQue,)).start()
        self.inference(self.imgQue, self.detQue, self.fpsQue, self.msgQue, self.stateQue)
                   
    def shutdown(self):
        self.RUNNING = False

    def imgCapture(self, imgQue, msgQue):
        msg = None
        while self.RUNNING:
            img = None
            try:
                msg = msgQue.get(block=False)
            except:
                pass
            if msg == False:
                print("Image Capture Process is shutting down")
                self.shutdown()
                msgQue.put(False)
                continue
            with mss.mss() as sct:
                monitor = {"top": 0, "left": 0, "width": self.initWidth, "height": self.initHeight}
                img = np.array(sct.grab(monitor))
        
            img = cv2.resize(img, (self.width, self.height),
                               interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                imgQue.put(img, timeout=3)
            except:
                print("img exception")
                pass

    def inference(self, imgQue, detQue, fpsQue, msgQue, stateQue):
        dImg = darknet.make_image(self.width, self.height, 3)
        msg = None
        names = {"Vayne":0, "Melee":1, "Ranged":2, "Seige":3}
        while self.RUNNING:
            prevTime = time.time()
            try:
                msg = msgQue.get(block=False)
            except:
                pass
            if msg == False:
                print("Inference Process is shutting down")
                self.shutdown()
                msgQue.put(False)
                continue
            try:
                img = imgQue.get(timeout=3)
            except:
                continue
            darknet.copy_image_from_bytes(dImg, img.tobytes())
            
            dets = darknet.detect_image(self.network, self.classNames,
                                        dImg, thresh=self.args.thresh)
            try:
                detQue.put(dets, timeout=3)
            except:
                 continue
            fps = int(1/(time.time() - prevTime))
            try:
                fpsQue.put(fps, timeout=2)
            except:
                continue
            print("FPS: {}".format(fps))
            obs = []
            for det in dets:
                classIdx = names[det[0]]
                prob = det[1]
                x = (det[2][0] + det[2][2])/2.0 * self.initWidth/self.width
                y = (det[2][1] + det[2][3])/2.0 * self.initHeight/self.height
                obs.append(Observation(classIdx, prob, x, y))
            print(obs)
            try:
                stateQue.put(obs, block=False)
            except:
                pass

    def drawing(self, imgQue, detQue, fpsQue, msgQue):
        random.seed(3)  # deterministic bbox colors
        self.window = cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Inference", 200, 2000)
        cv2.resizeWindow("Inference", self.height, self.width)
        while self.RUNNING:
            print("drawing")
            frame = imgQue.get()
            dets = detQue.get()
            fps = fpsQue.get()
            if frame is not None:
                print(dets)
                image = darknet.draw_boxes(dets, frame, self.classColors)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.height, self.width))
                #if self.args.display:
                cv2.imshow('Inference', image)
                if cv2.waitKey(fps) == 27:
                    print("shutting down")
                    msgQue.put(False)
                    self.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parser()
    obs = Observer(args)
    #obs.vidCapture1()
    obs.start()
    print("Observer is shutting down")
    sys.exit()
