import cv2 as cv
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0,trackCon=0):
        self.mode=mode
        self.maxHands=maxHands 
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList=[]


    def findHands(self,img,draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:# tracking each hand
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)

        return img       
                

    def findPosition(self,img,handNo=0,draw=True):
        self.lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]

            for id,lm in enumerate(myHand.landmark):
                    # print(id,lm)
                    h,w,c=img.shape # height width channel of img
                    cx,cy=int(lm.x*w),int(lm.y*h)# position 
                    # print(id,cx,cy)  # print id also for location for that id
                    self.lmList.append([id,cx,cy])
                    if draw:
                        cv.circle(img,(cx,cy),10,(255,0,255),cv.FILLED)

                    # if id==0:  # draw circle at landmark first
        return self.lmList
        
    def fingersUp(self):
        fingers = []
        # Thumb
        if len(self.lmList)!=0:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
    
        # Fingers
        for id in range(1, 5):
            if len(self.lmList)!=0:
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
    
            # totalFingers = fingers.count(1)
    
        return fingers
    
    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
    
        return length, img, [x1, y1, x2, y2, cx, cy]

        
    # cTime=time.time()
    # fps=1/(cTime-pTime)
    # pTime=cTime
    # # w=input("enter")
    # # if w==0:
    # #     break

    

    # cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    # cv.imshow('Image',img)
    # cv.waitKey(1)
    

def main():  # dummy code can use in different project
    
    pTime=0 # previous time
    cTime=0 # current time
    cap=cv.VideoCapture(0)
    detector=handDetector()

    while True:
        success,img=cap.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)!= 0:
            print(lmList[4])
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv.putText(img,f'FPS:{str(int(fps))}',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3) # insert text

        cv.imshow('Image',img)
        cv.waitKey(1)
   


if __name__== "__main__":
    main()