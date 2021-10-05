import cv2
import numpy as np 
import time
import HandTrackingModule as htm
import autopy


pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

#TO fix width and height of video
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

#Creating object of previous class
detector = htm.handDetector()

#Getting screen size
wScr, hScr = autopy.screen.size()

#Mouse was flickering too much to avoid that we have used smoothening
frameR = 70 # Frame Reduction
smoothening = 5

while True:
    success, img = cap.read() #Read Video from camera
    img = detector.findHands(img, True) #Find Hands in it
    lmList = detector.findPosition(img, draw=False) #Find Position in it
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp() #Check which fingers are up

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR,wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR,hCam-frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(wScr-clocX, clocY)
            cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        if fingers[1]==1 and fingers[2]==1:
            length, img, lineInfo = detector.findDistance(8,12, img)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0,255,0), cv2.FILLED)
                autopy.mouse.click()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1) 