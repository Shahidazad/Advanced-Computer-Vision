import cv2 as cv
import mediapipe as mp
import time 
import os
import handTrackingModule as htm

wcam,hcam=1700,1980
cap=cv.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)

folderpath='handTrackingproject/img'
myList=os.listdir(folderpath)
print(myList)
overlayList=[]
for imPath in myList:
    image=cv.imread(f'{folderpath}/{imPath}')
    print(f'{folderpath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

pTime=0
detector=htm.handDetector(detectionCon=0)
tipid=[4,8,12,16,20]
while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img)
    # print(lmList)
    if len(lmList)!=0:
        fingers=[]

        # for thumb
        if lmList[tipid[0]][1]>lmList[tipid[0]-1][1]:  # thumb is close or not
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if lmList[tipid[id]][2]<lmList[tipid[id]-2][2]:  # hand is close or not
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalfinger=fingers.count(1)
        print(totalfinger)

            
                # print('index finger open')
        h,w,c=overlayList[totalfinger-1].shape
        img[0:h,0:w]=overlayList[totalfinger-1]
        cv.putText(img,str(totalfinger),(270,680),cv.FONT_HERSHEY_PLAIN,9,(0,255,0),20)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(img,f'FPS:{int(fps)}',(400,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,0))


    cv.imshow('img',img)
    cv.waitKey(1)
