


class Simple_Vayne():
    def __inti__(self, observer = None, controller = None, reward=None):
        self.observer = observer
        self.controller = controller
        self.vayne = None
        print("Initializing Simple Vayne AI Controller")

    def getObs(self):
        self.obs = self.observer.stateQue.get()

    def updateBelief(self):
        for ob in self.obs:
            if ob.idx = 0:
                self.vayne = ob
        self.obs.del(self.vayne)
        

    def getDist(self, ob):
        dx = ob.x - self.vayne.x
        dy = ob.y - self.vayne.y

        return( dx * dx + dy * dy )

        
    def getActions(self):
        actions = []
        for ob in obs:
            actions.append( (self.getDist(ob), ('attack-click', (ob.x, ob.y)) ) )

    def selectAction(self, actions):
        if(len(actions) == 0):
            return(-1)
        miny = actions[0][0]
        bestAct = actions[0][1]
        for act in actions:
            if act[0] < miny:
                bestAct = act[1]
                miny = act[0]
        return bestAct
                            

    def mainLoop(self):
        while self.RUNNING:
            self.getObs()
            self.updateBelief()
            actions = self.getActions()
            bestAction = self.selectAction(actions)
            if bestAction == -1:
                continue
            self.controller.act(bestAction)
            
            


if __name__ == "__main__":
    print("Simple Vayne Controller coming soon to a computer near you")

    
