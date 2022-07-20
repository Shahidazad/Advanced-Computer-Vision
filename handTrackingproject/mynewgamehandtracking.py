
import cv2 as cv
import mediapipe as mp
import time
import handTrackingModule as htm

pTime=0 # previous time
cTime=0 # current time
cap=cv.VideoCapture(0)
detector=htm.handDetector()

while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img)
    if len(lmList) !=0:
        print(lmList[4])# it show landmark num 4
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),3) # insert text

    cv.imshow('Image',img)
    cv.waitKey(1)
   