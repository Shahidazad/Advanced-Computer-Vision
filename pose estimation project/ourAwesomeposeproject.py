import cv2 as cv
import time
import PoseModule as mp
  
  
cap=cv.VideoCapture('pose estimation project/videos/v.mp4')
pTime=0
    # mpDraw=mp.solutions.drawing_utils
detector=mp.poseDetector() 
while True:
    success,img=cap.read()
    img=detector.findpose(img)
    lmList=detector.findposition(img)
    if len(lmList)!=0:
        print(lmList[14])
        cv.circle(img,(lmList[14][1],lmList[14][2]),10,(255,0,0),-1)
    cv.imshow('img',img)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    

    cv.putText(img,str(int(fps)),(70,50),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv.imshow('img',img)
    cv.waitKey(2)
