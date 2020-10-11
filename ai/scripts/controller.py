
from pynput import *
import time

class Controller():
    def __init__(self):
        self.mouse = mouse.Controller()
        self.keyboard = keyboard.Controller()

    def act(self, action):
        actionType = action[0]
        actionData = action[1]
        if actionType == "Attack_Click":
            self.attackClick(actionData)
        if actionType == "Move_Click":
            self.moveClick(actionData)
    
    def attackClick(self, pos):
        self.mouse.position = pos
        self.keyboard.press('a')
        self.mouse.press(mouse.Button.left)
        self.mouse.release(mouse.Button.left)
        self.keyboard.release('a')

    def moveClick(self, pos):
        self.mouse.position = pos
        self.mouse.press(mouse.Button.left)
        self.mouse.release(mouse.Button.left)
        


if __name__=="__main__":
    print("Controller online")
    con = Controller()
    a =  ['Attack_Click', (300, 400) ]
    b =  ['Attack_Click', (1920 - 700, 1080 - 500)]
    while True:
        con.act(a )
        time.sleep(5)
        con.act(b)
        time.sleep(5)
    
