
import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0,trackCon=0):
        self.mode=mode
        self.maxHands=maxHands 
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils


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
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]

            for id,lm in enumerate(myHand.landmark):
                    # print(id,lm)
                    h,w,c=img.shape # height width channel of img
                    cx,cy=int(lm.x*w),int(lm.y*h)# position 
                    # print(id,cx,cy)  # print id also for location for that id
                    lmList.append([id,cx,cy])
                    if draw:
                        cv.circle(img,(cx,cy),10,(255,0,255),cv.FILLED)

                    # if id==0:  # draw circle at landmark first
        return lmList
        


        
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