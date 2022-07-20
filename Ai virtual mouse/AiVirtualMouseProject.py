import cv2 as cv
import time
import numpy as np
import handTrackingModule as htm
import autopy


wCam,hCam=640,480
frameR=100 # frame reduction
smoothening=7
plocX,plocY=0,0
clocX,clocY=0,0


cap=cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)


pTime=0
wScr,hScr=autopy.screen.size()  # size of screen
print(wScr,hScr)

detector=htm.handDetector(maxHands=1)
while True:
    # 1.find hand landmark
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img)

    
    
    #2. get tip of index and middle finger
    if len(lmList)!=0:
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:] 

        # print(x1,y1,x2,y2)

        # 3. check which finger r up
        fingers=detector.fingersUp()
        print(fingers)
        cv.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
        # 4.only index finger : moving mode
        if fingers[1]==1 and fingers[2]==0:
             #5 convert Cordinate
            x3=np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            # 6. smooth value
            clocX=plocX +(x3-plocX)/smoothening
            clocY=plocY +(y3-plocY)/smoothening
    
           # 7. move mouse
            autopy.mouse.move(wScr - clocX,clocY)
            cv.circle(img,(x1,y1),15,(255,0,0),-1)
              
            plocX,plocY=clocX,clocY
    # 8. both index and middle finger r up then it is clicking mode
        if fingers[1]==1 and fingers[2]==1:
            # 9. find distance bet fingers
            length,img,lineinfo=detector.findDistance(8,12,img)
        # 10 click mouse if distance short
            if length <40:
                cv.circle(img,(lineinfo[4],lineinfo[5]),15,(255,0,0),-1)
                autopy.mouse.click()
    
    
    # 11. Frame Rate

    cTime=time.time()
    fps=1/(cTime-pTime)
    cTime=pTime

    cv.putText(img,str(int(fps)),(20,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv.waitKey(1)