from numpy import load
import numpy as np
from cv2 import *
from math import fabs
fileName = "../DeepLeagueData/clusters_cleaned/test/data_test_set_cluster_0.npz"
data = load(fileName, allow_pickle=True)

items = data.files
imgs = data['images']
boxes = data['boxes']



def draw_boxes(img, boxes, box_classes=None, class_names=None, scores=None):
    """Draw bounding boxes on image.
    Draw bounding boxes with class name and optional box score on image.
    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.
    Returns:
        A copy of `image` modified with given bounding boxes.
    """
    boxImg = img.copy()
    for box in boxes:
        print box
        (label, minX, minY, maxX, maxY) = box
        cv2.rectangle(boxImg, (minX, minY), (maxX, maxY), (0,0,255))

    return boxImg
# 7-281  274
# 7-281  274
# going dark = -100, -100, -90
def process(image):
    
    """
    An image of minimap ---> a numpy array of champions image with dimension 24*24
    """
    champion_list = []
    coords = []
    b,g,r = cv2.split(image)
    inranger = cv2.inRange(r,120,255)
    inrangeg = cv2.inRange(g,120,255)
    inrangeb = cv2.inRange(b,120,255)
    induction = inranger - inrangeg - inrangeb

    circles = cv2.HoughCircles(induction,cv2.HOUGH_GRADIENT,1,10,param1 = 20,param2 =10,minRadius = 9, maxRadius = 15)
    if(circles is not None):
        for n in range(circles.shape[1]):
            x = int(circles[0][n][0])
            y = int(circles[0][n][1])
            coords.append([x,y])
            radius = int(circles[0][n][2])
            cropped = image[y-radius:y+radius,x-radius:x+radius].copy()
            print "Size:", int(2 * radius)
            #to_append = cv2.resize(cropped,(24,24))
            champion_list.append(cropped)
            cv2.rectangle(image,(x-radius,y-radius),(x+radius,y+radius),(255,255,255),1)
        #champion_list = np.stack(champion_list,axis = 0,)
        #champion_list = champion_list.reshape((champion_list.shape[0], 24, 24, 3))
        #champion_list_text = league_scanner.predict(champion_list)
        #for n in range(len(champion_list_text)):
        #    cv2.putText(image,class_names[champion_list_text[n]],(coords[n][0]-12,coords[n][1]-12), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
    else:
        image = image
    return image, champion_list

def hPlay(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #r,g,b = cv2.split(img)
    #iRR = cv2.inRange(r,40,200)
    #iRG = cv2.inRange(g,20,100)
    #iRB = cv2.inRange(b, 20, 50)
    #cImg = iRR - iRG - iRB
    cImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #gImg = cv2.medianBlur(gImg, 5)
    cv2.imshow("red", cImg)
    circles = cv2.HoughCircles(cImg, cv2.HOUGH_GRADIENT,1,10, param1=100, param2=10,
                               minRadius=8, maxRadius=15)
    if circles is None:
        print "Found no circles"
        return img
    for k, i in enumerate(circles[0,:]):
        cv2.circle(img, (i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow("Detected Circles", img)

def near(pt, bgPt):
    db = abs(int(pt[0]) - int(bgPt[0]))
    dg = abs(int(pt[1]) - int(bgPt[1]))
    dr = abs(int(pt[2]) - int(bgPt[2]))
    
    db2 = abs(int(pt[0]) - int(bgPt[0]) +  90)
    dg2 = abs(int(pt[1]) - int(bgPt[1]) + 100)
    dr2 = abs(int(pt[2]) - int(bgPt[2]) + 100)

    dr3 = abs(int(pt[2]) - int(bgPt[2]) + 30)
    maxDif = 20
    return (db < maxDif and dr < maxDif and dg < maxDif) or (db2 < maxDif and dr2 < maxDif and dg2 < maxDif) or (db2 < maxDif and dr3 < maxDif and db2 < maxDif)

def removeBG(img, bg):
    filt = img.copy()
    mask = np.ones((271,271,1), dtype = "uint8") * 255
    for u in range(271):
        for v in range(271):
            bgPt = bg[u,v]
            pt = img[u,v]
            if near(pt, bgPt):
                mask[u,v] = 0
                filt[u,v,0] = 255
                filt[u,v,1] = 255
                filt[u,v,2] = 255
                
    #mask = np.zeros((274,274),dtype="uint8")
    # zero is black "i.e black means the same"
    return mask, filt


def redCircle(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tet1 = cv2.inRange(img, (0, 0, 0), (20, 255, 255))
    tet2 = cv2.inRange(img, (140,0,0), (180,255,255))
    tet = cv2.bitwise_or(tet1, tet2)
    #tet = cv2.medianBlur(tet, 5)
    #tet = cv2.GaussianBlur(tet, (5,5), 0)
    #tet = cv2.bilateralFilter(tet, 9,75,75)
    #print cv2.cvtColor(np.uint8([[[75,35,80]]]), cv2.COLOR_BGR2HSV)
    #cv2.imshow("Thesh", tet)
    
    masked = cv2.bitwise_or(img,img, mask=tet)
    masked = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)
    cv2.imshow("Masked", masked)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    circles = cv2.HoughCircles(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY),
                               cv2.HOUGH_GRADIENT,1,10, param1=40, param2=15,
                               minRadius=8, maxRadius=18)
    if circles is None:
        print "Found no circles"
        return img
    for k, i in enumerate(circles[0,:]):
        cv2.circle(img, (i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow("Detected Circles", img)
if __name__ == "__main__":
    #bg = cv2.imread("map11.png")
    #bg = cv2.resize(bg, (274,274))
    #bg = bg[1:272,1:272]
    #bg = cv2.bilateralFilter(bg, 5, 75, 75)

    #backSub = cv2.createBackgroundSubtractorMOG2()

    #for img in imgs:
        #fgMask = backSub.apply(img)
        #cv2.imshow("Mask", fgMask)
        #cv2.waitKey(1)
    for i, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mkImg, champList = process(img)

        cv2.imshow("Masked Image", mkImg)
        print "Fragements found:", len(champList)
        for k, champImg in enumerate(champList):
            cv2.imshow("Fragment " + str(k), champImg)
        cv2.waitKey(0)
        
        #img2 = img#[8:280, 8:280]
        #img2 = cv2.bitwise_or(img2, img2, mask = backSub.apply(img))
        #img2 = cv2.bilateralFilter(img2, 5,75, 75)
        #mask, filt = removeBG(img2, bg)
        #hPlay(img2)
        #redCircle(img2)
        #boxImg, champList = process(img)
        #filt = cv2.cvtColor(filt, cv2.COLOR_RGB2BGR)
        #boxImg = draw_boxes(img, boxes[i])
        #cv2.imshow("Original", img)
        #cv2.imshow("Cropped", img2)
        #cv2.imshow("Filtered", filt)
        #cv2.imshow("Mask", mask)
        #cv2.imshow("With Boxes", boxImg)
        #for i, crop in enumerate(champList):
        #    cv2.imshow("Crop:" + str(i), crop)
        #cv2.waitKey(0)




        
        if i == 4:
            break
    cv2.destroyAllWindows()
