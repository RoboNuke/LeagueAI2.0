import heapq
from time import *
from controller import Controller
from simple_observer import *

class Simple_Vayne(object):
    def __init__(self, observer=None, controller=None, reward=None, defaultPose=None):
        self.observer = observer
        self.controller = controller
        self.vayne = None
        self.seeVayne = False
        self.RUNNING = True

        if defaultPose == None:
            self.defaultPos = (300, 1080-300)
        else:
            self.defaultPos = defaultPose
        print("Initializing Simple Vayne AI Controller")

    def getObs(self):
        self.obs = self.stateQue.get()

    def updateBelief(self):
        #print(self.obs)
        if len(self.obs) == 0:
            return
        for ob in self.obs:
            if ob.idx == 0:
                self.vayne = ob
        if self.vayne in self.obs:
            self.obs.remove(self.vayne)
            self.seeVayne = True

        #print("Belief:", self.obs)
        

    def getDist(self, ob):
        dx = ob.x - self.vayne.x
        dy = ob.y - self.vayne.y

        return( dx * dx + dy * dy )

        
    def getActions(self):
        actions = []
        if not self.seeVayne: # if we can't see vayne then move to a set location in the image
            heapq.heappush(actions, (-1, ('Move_Click', self.defaultPos)))
        if self.vayne == None:
            return actions
        for ob in self.obs:
            heapq.heappush(actions, (self.getDist(ob), ('Attack_Click', (ob.x, ob.y))))
        #print("Actions:", actions)
        return actions

    def selectAction(self, actions):
        
        if(len(actions) == 0):
            return(-1)
        
        return heapq.heappop(actions)[1]
    
    def shutdown(self):
        self.RUNNING = False
        
    def mainLoop(self, stateQue, msgQue):
        msg = None
        self.stateQue = stateQue
        while self.RUNNING:
            try:
                msg = msgQue.get(block=False)
                print("Got Msg")
            except:
                pass
            if msg == False:
                print("Vayne AI Shutting down")
                msgQue.put(False)
                self.shutdown()
                continue
            self.getObs()
            if len(self.obs) == 0:
                continue
            self.updateBelief()
            if len(self.obs) == 0:
                continue
            actions = self.getActions()
            bestAction = self.selectAction(actions)
            print("Best Action:", bestAction)
            if bestAction == -1:
                continue
            self.controller.act(bestAction)
            
            


if __name__ == "__main__":
    #print("Simple Vayne Controller coming soon to a computer near you")
    time.sleep(3)aa
    args = parser()
    con = Controller()
    obs = Observer(args)
    
    
    vayne = Simple_Vayne(obs, con)
    
    v = multiprocessing.Process(target=vayne.mainLoop, args=(obs.stateQue, obs.msgQue,)).start()
    obs.start()
    print("Why dave why")
    
    
