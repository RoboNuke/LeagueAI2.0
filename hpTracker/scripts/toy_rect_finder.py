import cv2
import imutils

def detect(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) == 4:
        (x,y,w,h) = cv2.boundingRect(approx)
        return("Rectangle")
    else:
        return(-1)

def getRects(frame):
    # preprocess img
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        shape = detect(c)
        if shape != -1:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.drawContours(frame, [c], -1, (0,255, 0), 2)
            cv2.putText( frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

    return(frame)

if __name__ == "__main__":
    img = cv2.imread("../data/shapes_and_colors.jpg")
    frame = getRects(img.copy())

    cv2.imshow("Final", frame)
    cv2.imshow("Initial",img)
    cv2.waitKey(0)
