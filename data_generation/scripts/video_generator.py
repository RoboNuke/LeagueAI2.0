from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
from pynput import *

import sys, getopt
import yaml

import cv2
import numpy as np
import pyautogui


"""
 For the animations   id: 'model-viewer-animation'
 For champs/creatures id: 'model-viewer-champions'
 For skins            id: 'model-viewer-champion-skins' # to-do
"""

# Global Variables

# System files    
configFile = "../cfg/vayne.yaml"
outputFile = "../data/1videos/"
prefix = ""

# Video Config
# display screen resolution, get it from your OS settings
SCREEN_SIZE = (1920, 1080)
# define the codec
fourcc = cv2.VideoWriter_fourcc(*"XVID")
POSITION = (0,1000)
MOVE = 5
MOVESTEPS = 100


def selectCreeps(driver):
    """ Will open the window for creatures """
    driver.get("http://www.teemo.gg/model-viewer?model-type=creatures")
    sleep(5)

def selectChamps(driver):
    """ Open window for champs """
    print "here"
    driver.get("http://www.teemo.gg/model-viewer?model-type=champions")
    print "here"
    sleep(5)

def getOptions(ide):
    """ Returns all the options that can be selected by an id """
    sel = driver.find_element_by_id(ide)
    return sel.find_elements_by_tag_name('option')

def load(name, options):
    """ Loads the option given (i.e for champions) """
    for option in options:
        if option.text == name:
            option.click()
            driver.find_element_by_id('model-viewer-load-button').click()
            sleep(10)
            break

def selectAnimation(name, animations):
    "Changes the animation for the currently loaded model """
    for option in animations:
        if option.text == name:
            option.click()
            break

def fullscreen(driver):
    """ Sets the model viewer window to fullscreen """
    driver.find_element_by_id('model-viewer-fullscreen').click()
    sleep(5)

def exitFullscreen(driver):
    """ When model viewer screen window fullscreen, this will minimize it again """
    k = keyboard.Controller()
    k.press(keyboard.Key.esc)
    k.release(keyboard.Key.esc)
    sleep(5)

def setViewpoint():
    """ Moves the camera angle to be simular to what it looks like in game """
    # Rotate the camera for top view
    m = mouse.Controller()
    m.position = (100, 200)
    m.press(mouse.Button.left)
    m.move(0,75)
    m.release(mouse.Button.left)
    m.scroll(0,-4)
    
def rotateRecordAnimation(vid):
    """ Slowly rotates the animation in a circle a few times """
    global POSITION, MOVE, MOVESTEPS
    m = mouse.Controller()
    m.position = POSITION
    for i in range(MOVESTEPS):
        m.press(mouse.Button.left)
        m.move(MOVE,0)
        m.release(mouse.Button.left)
        m.position = POSITION
        # make a screenshot
        img = pyautogui.screenshot()
        # convert these pixels to a proper numpy array to work with OpenCV
        frame = np.array(img)
        # convert colors from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # write the frame
        vid.write(frame)


def unpack(argv):
    global outputFile, configFile, prefix
    try:
        opts, args = getopt.getopt(argv,"hc:o:p:", ["configFile=", "outputFile=", "prefix="])
    except getopt.GetoptError:
        print 'video_generator.py -c <configuration file> -o <output file> -p <prefix>'
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            print  'web_interact.py -c <configuration file> -o <output file> -p <prefix>'
            sys.exit()
        elif opt in ("-c", "configFile="):
            configFile = arg
            print "here"
        elif opt in ("-o","outputFile="):
            outputFile = arg
        elif opt in ("-p","prefix="):
            prefix = arg
            
    todoList = yaml.load(open(configFile), Loader = yaml.FullLoader)
    champs = todoList["champions"]
    creeps = todoList["creatures"]
    print "Will film the following champions:"
    for c in champs:
        print "--", c
    print "Will film the following creatures:"
    for c in creeps:
        print "--", c
    print "Their videos can be found at: ", outputFile
    print "All videos will be prefixed with: ", prefix
    return (champs, creeps)
            
def createVids(desired, options, driver):
    global fourcc, SCREEN_SIZE, outputFile, prefix
    for option in desired:
        # create the video write object
        vid = cv2.VideoWriter(outputFile + prefix + option + ".avi", fourcc, 20.0, (SCREEN_SIZE))
        try:
            load(option, options)
        except:
            print "Champion or Creatures ", option, " is not an option"
            print "Continuing..."
            continue
        animations = getOptions("model-viewer-animation") #get all possible animations
        fullscreen(driver) # set window to full screen
        setViewpoint()
        for animation in animations: 
            print(animation.text)
            #skip empty animations
            if(animation.text == "Animation" or animation.text == "No Animation"):
                continue
            selectAnimation(animation.text, animations)
            rotateRecordAnimation(vid)
            
        vid.release()
    exitFullscreen(driver)
        
        
if __name__ == "__main__":


    # Start Chrome full screen
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(chrome_options = options)

    # unpack config info
    (champs, creeps) = unpack(sys.argv[1:])
    # Start with champion videos
    selectChamps(driver)
    options = getOptions("model-viewer-champions")  #get list of champs
    if champs[0] == "All":
        createVids(options, options, driver)
    else:
        createVids(champs, options, driver)

    # Finish with creatures
    selectCreeps(driver)
    options = getOptions("model-viewer-champions") # get full creature list
    if creeps[0] == "All":
        createVids(options, options, driver)
    else:
        createVids(creeps, options, driver)
        
    driver.close()
    
