import argparse
import cv2
import heapq
import numpy as np
import matplotlib.pyplot as plt
import copy
labelColor={ 0: (255,0,0),
             1: (0,0,255),
             2: (0,255,0),
             3: (255,0,255)}

guessLabelColor = {  0: (255,50,50),
                     1: (50,50,255),
                     2: (50,255,50),
                     3: (255,50,255)}
def arg_parse():

    parser = argparse.ArgumentParser(description='Data Evaluation for Yolo using mAP & IoU')
    parser.add_argument('-r', dest = 'r', help = "Path to results file for evaluation", default="results/vayne/")
    parser.add_argument('-t', dest = 't', help = "Path to the dataset file", default="data/vayneDataset/")
    parser.add_argument('-o', dest = 'o', help = "Path to where graphs/evaluation should be saved", default = 'None')
    parser.add_argument('-n', dest = 'n', help = 'Path to .names file', default = "data/vayneDataset/vayne.names")

    parser.add_argument('-s', dest='SHOW',help = 'Show precision vs recall curves', default = 'False')

    return parser.parse_args()
"""
def max(a, b):
    if a > b:
        return a
    return b

def min(a,b):
    if a > b:
        return b
    return a
"""
def calcIOU(a, b):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    boxBArea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def getNames(filepath):
    f = open(filepath, 'r')
    names = f.read().split('\n')[:-1]
    return names


def getGuesses(name, resultFile):
    f = open(resultFile + "comp4_det_test_"+ name + '.txt', 'r')
    guesses = {}
    picData = []
    first = True
    picName = ''
    for line in f.readlines(): 
        dat = line.strip().split(' ')
        if picName != dat[0]:
            guesses[picName] = picData
            picData = []
            picName = dat[0]
            
        picData.append([dat[1], convToPts([float(x) for x in dat[2:]])])
    f.close()
    return(guesses)


def getLines(lines, picName):
    data = []
    for line in lines:
        dat = line.strip().split(' ')
        if picName == dat[0]:
            data.append([dat[1], [float(x) for x in dat[2:]]])
    return data


def getGus(resultFile, picNames, nameDic):
    files = [open(resultFile +  "comp4_det_test_"+ nameDic[x] + '.txt', 'r').readlines() for x in range(len(nameDic))]
    data = {}
    #count = 0
    for picName in picNames:
        #count += 1
        #if count >5:
        #    break
        dat = {}
        for i in range(len(files)):
            dat[nameDic[i]] = getLines(files[i], picName)
        data[picName] = dat
    return data
    
def convToPts(bbox,pw = 1.0, ph = 1.0):
    x,y,w,h = bbox
    return([(x-w/2)*pw, (y-h/2)*ph, (x+w/2)*pw, (y+h/2)*ph])

def getGroundTruth(truthFile, nameDic):
    f = open(truthFile + 'test.list', 'r')
    lines = f.readlines()
    f.close()
    gTData = {}
    #count = 0
    for line in lines:
        #count += 1
        #if count >5:
        #    break
        pat = line.strip()
        img = cv2.imread(pat)
        h,w,c = img.shape
        w = w - 4
        #print(pat[:-4] + '.txt')
        k = open(pat[:-4] + '.txt', 'r')
        picName = pat.split('/')[-1][0:-4]
        picData = {}
        for n in nameDic.values():
            picData[n] = []
        for subLine in k.readlines():
            dat = subLine.strip().split(' ')
            picData[nameDic[int(dat[0])]].append(convToPts([float(x) for x in dat[1:]],w,h))
        gTData[picName] = picData
        k.close()
    return(gTData)

def argmax(x):
    return x.index(max(x))
        

def evaluate(gts, guesses, nameDic, iouTh = 0.5, classProb=0.0):
    # each [] a heap with -prob as key and true if TP or false if FP as element
    # one [] for each class
    guesses = copy.deepcopy(guesses)
    cor = [[] for x in range(len(nameDic))]
    fN = [0 for x in range(len(nameDic))]
    # {picName:{class name:[[xmin, ymin, xmax, ymax], ...]}}
    #{pic name:{class name: [ [prob, [xmin ...]], ...]
    for picName in gts.keys():
        #print(picName)
        for i in range(len(gts[picName].keys())):
            picBBoxs = gts[picName][nameDic[i]]
            gus = guesses[picName][nameDic[i]]
            ious = []
            for obj in picBBoxs: # iterate over all bboxes of this class to find best match
                for bb in gus:
                    if float(bb[0]) > classProb:
                        tIOU = calcIOU(obj, bb[1])
                        if tIOU > iouTh: # represents a positive result
                            ious.append(calcIOU(obj, bb[1]))
                        else:
                            ious.append(-1)
                    else:
                        ious.append(-1)
                if( len(ious) == 0):
                    fN[i] += 1
                    ious = []
                    continue
                best = argmax(ious)
                if( ious[best] == -1 ):
                    fN[i] += 1
                    ious = []
                    continue

                for k,u in enumerate(ious):
                    if u == -1:
                        #print("\tContinued")
                        continue # represents not a guess (not FP or TP)
                    if u == ious[best]:
                        #print("\tFound Best")
                        heapq.heappush(cor[i], (-float(gus[k][0]), u, True))
                        fN[i] = fN[i] +  1
                    else:
                        #print("\tNormal")
                        heapq.heappush(cor[i], (-float(gus[k][0]), u, False))
                del gus[best]
                ious = []
    return (cor, fN)
                        
                    
            
            

def drawRect(frame, pts, labe, guess=False):
    global labelColor, guessLabelColor
    if not guess:
        cv2.rectangle(frame, (int(pts[0])+2, int(pts[1])),
                      (int(pts[2]), int(pts[3])), labelColor[labe], 1)
    else:
        cv2.rectangle(frame, (int(pts[0])+2, int(pts[1])),
                      (int(pts[2]), int(pts[3])), (255,255,255), 1)
        
    return frame

def disResults(gts, guesses, truthFile, names):
    for picName in gts.keys():
        img = cv2.imread(truthFile + 'test/' + picName + '.jpg')
        for name in gts[picName].keys():
            for pts in gts[picName][name]:
                drawRect(img, pts, names.index(name))
            #print(guesses[picName])
            for pts in guesses[picName][name]:
                drawRect(img, pts[1], names.index(name), True)
        cv2.imshow(picName, img)
        key = cv2.waitKey(1)
        cv2.destroyAllWindows()
        if key == 27:
            sys.exit()
    
def smoothPre(pre):
    pre.reverse()
    maxy = pre[0]
    for i in range(len(pre)):
        if pre[i] < maxy:
            pre[i] = maxy
        else:
            maxy = pre[i]
    pre.reverse()
    return pre
def getData(cor, totPs):
    pres = []
    recs = []
    tPs = []
    fPs = []
    fNs = []
    for i in range(len(totPs)):
        pre = []
        rec = []
        tP = 0
        fP = 0
        pT = 0.0
        d = 0.0
        recall = 0.0
        for k in range(len(cor[i])):
            x = heapq.heappop(cor[i])
            if x[2]:
                pT = pT + 1.0
                tP = tP + 1
            else:
                fP = fP + 1
            d = d + 1.0
            pre.append(pT/d)
            rec.append(pT/totPs[i])
        tPs.append(tP)
        fPs.append(fP)
        fNs.append(totPs[i] - tP)
        pres.append(pre)
        recs.append(rec)
    return(recs, pres, tPs, fPs, fNs)
    
def calcmAP(pre, rec):
    h = pre[0]
    w = rec[0]
    mAP = 0.0
    for i in range(len(pre)):
        if h == pre[i]:
            continue
        else:
            mAP = mAP + h * (rec[i-1] - w)
            w = rec[i]
            h = pre[i]
    return(mAP)

def calcF1(tP, fP, totP):
    pre = tP/(tP+fP)
    rec = tP/totP
    return 2*((pre*rec)/(pre+rec))
        
if __name__ == "__main__":
    args = arg_parse()

    SHOW=False
    if args.SHOW == 'True':
        SHOW = True
    resultFile = args.r
    truthFile = args.t
    outFile = args.o
    nameFile = args.n
    if outFile == 'None':
        outFile = resultFile
    
    names = getNames(nameFile)
    #names = ["Vayne"]
    nameDic = {}
    for i in range(len(names)):
        nameDic[i] = names[i]

    # unpack the data
    gtd = getGroundTruth(truthFile, nameDic) # {picName:{class name:[xmin, ymin, xmax, ymax]}}
    guesses = getGus(resultFile, gtd.keys(), nameDic) #{pic name:{class name: [ [prob, [xmin, ... 

    #disResults(gtd, guesses, truthFile, names)
    
    (cor, totPs) = evaluate(gtd, guesses, nameDic)
    
    (recs, pres, tPs, fPs, fNs) = getData(cor, totPs)

    smPres = []
    for pre in pres:
        smPres.append(smoothPre(copy.deepcopy(pre)))
    mAPs = []
    for i in range(len(pres)):
        mAP = calcmAP(smPres[i], recs[i])
        mAPs.append(mAP)
        fig =  plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(recs[i],pres[i], label='original')
        plt.plot(recs[i],smPres[i], label='smoothed')
        plt.title("Precision Vs Recall for " + names[i])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        ax.text(0.5, 0.01, "mAP=" + '{:.1f}'.format(mAP*100.0)+"%",
                 verticalalignment='bottom', horizontalalignment='left',
                 transform=ax.transAxes,
                 color='black', fontsize=12)
        plt.legend()
        if(SHOW):
            plt.show()
        plt.savefig("results/vayne/" + names[i] + "_precision_v_recall.png", bbox_inches='tight')

    """ Analysis on use case """
    
    (cor, totPs) = evaluate(gtd, guesses, nameDic, 0.5, 0.1)
    
    (recs, pres, tPs, fPs, fNs) = getData(cor, totPs)
    fig = plt.figure()
    plt.title('Confusion Chart per Class (IoU=0.5)')
    ind = [ x for x in range(len(names))]
    width = 0.35
    p1 = plt.bar(ind, tPs, width)
    p2 = plt.bar(ind, fNs, width, bottom = tPs)
    p3 = plt.bar(ind, fPs, width, bottom = [tPs[i]+fNs[i] for i in range(len(tPs))])
    plt.ylabel('# objects being classified')
    plt.xlabel('Class')
    plt.xticks(ind, names)
    plt.legend((p1,p2,p3), ('True Positives with IoU=0.5', 'False Negatives', 'False Positives'))
    if(SHOW):
        plt.show(fig)
    plt.savefig('results/vayne/confBar.png')

    fig = plt.figure()
    plt.title('mAP for Classes (IoU=0.5)')
    plt.bar(ind, mAPs, width)
    plt.ylabel("mAP")
    plt.xlabel('Class')
    plt.xticks(ind, names)
    if(SHOW):
        plt.show(fig)
    plt.savefig('results/vayne/mAP_bar.png')

    fig = plt.figure()
    plt.title('F1 for Classes (IoU=0.5)')
    plt.bar(ind, [calcF1(tPs[i], fPs[i], totPs[i]) for i in range(len(tPs))], width)
    plt.ylabel("F1 Score")
    plt.xlabel('Class')
    plt.xticks(ind, names)
    if(SHOW):
        plt.show(fig)
    plt.savefig('results/vayne/F1_bar.png')
    
    
