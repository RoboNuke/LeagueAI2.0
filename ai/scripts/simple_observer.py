

class State():
    def __init__(self, champs = None, creeps = None):
        self.champs = champs
        self.creeps = creeps
        


class Observer(object):
    def __init__(self):
        print("Initializing Observer")
        self.RUNNING = True
        
    def setImg(self, newImg):
        self.img = newImg.copy()
        self.newImg = True

    def shutdown(self):
        self.RUNNING = False

    def preprocess(self, frame):
        return(frame)

    def extractBBoxs(self, frame):
        """ Returns vayne's bbox and the bbox of all creeps in view """
        pass

    def setObs(self, bboxs):
        """ Returns a State object containing the clickable location for each object in view """
        self.obs = State()

    def getObs(self):
        return self.getObs
    
    def mainLoop(self):
        while self.RUNNING:
            if not self.newImg:
                continue
            frame = self.img
            self.newImg = False

            


if __name__ == "__main__":
    print("Observer coming soon to a computer near you")
