


class Simple_Vayne():
    def __inti__(self, observer = None(), reward=None):
        self.observer = observer
        print("Initializing Simple Vayne AI Controller")

    def getObs(self):
        self.obs = self.observer.getObs()
        self.newObs = True

    def selectAction(self, actions):
        pass

    def updateBelief(self):
        pass

    def getActions(self):
        pass

    def mainLoop(self):
        while self.RUNNING:
            if not self.newObs:
                continue
            self.newObs = False

            self.updateBelief(self.obs)
            actions = self.getActions
            self.action = self.selectAction(actions)
            
            


if __name__ == "__main__":
    print("Simple Vayne Controller coming soon to a computer near you")

    
