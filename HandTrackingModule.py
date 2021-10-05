# Importing Libraries for future use
import cv2
import mediapipe as mp
import time
import math

class handDetector():
    #When Class's object is created then this init constructor is called
    def __init__(self):
        #initializing objects to find hands and tools to draw on hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
    
    #To find Hands in Video
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert it to RGB to detect hands properly
        self.results = self.hands.process(imgRGB) # To actually find hands

        #Find hand Landmarks from Hand
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) 
        return img
    
    # Find Position of points
    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            self.myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(self.myHand.landmark): 
                h, w, c = img.shape # Fin Width and Height of Images/video  
                cx, cy = int(lm.x*w), int(lm.y*h) # Find position of point on that image
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
        return self.lmList
    
    #To find Which fongers are up
    def fingersUp(self):
        tipIds = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if self.lmList[tipIds[0]][2] > self.lmList[tipIds[0] - 1][2]:
                fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    #To find distance between 2 fingers that is forefinger and middlefinger
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1+ x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        return length, img, [x1, y1, x2, y2, cx, cy]