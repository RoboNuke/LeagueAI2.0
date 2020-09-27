

def parser():
    parser = argparse.ArgumentParser(description="Simple Vayne State Observer")

    parser.add_argument("--cfg", type=str, default="cfg/yolov3-vayne.cfg",
                        help="path to network configuration file")

    parser.add_argument("--weights", type=str, default="backup/vayne/yolov3-vayne_last.weights",
                        help="path to network pre-trained weight file")

    parser.add_argument("--display", type=bool, default=True,
                        help="Set to true to create a second window displaying the state")

    parser.add_argument("--data_file", type=str, default="data/vaynedataset/vayne.data",
                        help = "path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")

class State():
    def __init__(self, champs = None, creeps = None):
        self.champs = champs
        self.creeps = creeps
        

class Observer(object):
    def __init__(self, args):
        print("Initializing Observer")
        self.args = args
        self.RUNNING = True

        #set up queues
        self.frameQueue = Queue()
        self.dImgQueue = Queue(maxSize=1)
        self.fpsQueue = Queue(maxsize=1)
        self.detQueue = Queue(maxsize=1)
        
        # set up network
        self.network, self.classNames, self.classColors = darknet.load_network(
            self.args.cfg,
            self.args.data_file,
            self.args.weights,
            batch_size=1
        )

        # create darknet img
        self.width = darknet.network_width(network)
        self.height = darknet.network_height(network)
        self.dImg = darknet.make_image(width, height, 3)
                                     
    def start(self):
        Thread(target=self.vidCapture).start()
        Thread(target=self.inference).start()
        Thread(target=self.drawing)
        
    def shutdown(self):
        self.RUNNING = False

    def vidCapture(self): # ready for testing
        while obs.RUNNING:
            frame = pyautogui.screenshot(region =
                                         (0, 0,
                                          1920,1080))
            frame = np.array(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.width, self.height-self.padding),
                                       interpolation=cv2.INTER_LINEAR)
            frame_padded = cv2.copyMakeBorder(frame_resized,
                                              self.padding/2, self.padding/2, 0, 0,
                                              cv2.BORDER_CONSTANT)
            self.frameQueue.put(frame_padded)
            darknet.copy_image_from_bytes(darknet_image, frame_padded.tobytes())
            self.dImgQueue.put(darknet_image)

    def inference(self): #ready for testing
        while obs.RUNNING:
            darknet_image = self.dImgQueue.get()
            prev_time = time.time()
            detections = darknet.detect_image(self.network, self.classNames,
                                              self.dImg, thresh=self.args.thresh)
            self.detQueue.put(detections)
            fps = int(1/(time.time() - prev_time))
            self.fpsQueue.put(fps)
            print("FPS: {}".format(fps))
            #darknet.print_detections(detections, args.ext_output)

    def drawing(self):
        random.seed(3)  # deterministic bbox colors
        while obs.RUNNING:
            frame_padded = self.frameQueue.get()
            detections = self.detQueue.get()
            fps = self.fpsQueue.get()
            if frame_padded is not None:
                image = darknet.draw_boxes(detections, frame_padded, self.classColors)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #if args.out_filename is not None:
                #    video.write(image)
                if args.diplay == 'True':
                    cv2.imshow('Inference', image)
                if cv2.waitKey(fps) == 27:
                    break
        cv2.destroyAllWindows()
        

            


if __name__ == "__main__":
    print("Observer coming soon to a computer near you")
