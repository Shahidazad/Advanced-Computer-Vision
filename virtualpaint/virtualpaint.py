import cv2 as cv
import time
import os
import numpy as np
import handTrackingModule as htm 

folderPath='virtualpaint/img'
myList=os.listdir(folderPath)
print(myList)


overlayList=[]
for imPath in myList:
    image=cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header=overlayList[0]
drawColor=(255,0,255) #initial value
cap=cv.VideoCapture(0)
cap.set(3,1280)  #width should be same
cap.set(4,720)   #height should be same
detector=htm.handDetector()
brushThickness=15
xp,yp=0,0 #initial value

imgCanvas=np.zeros((720,1280,3),np.uint8)  # new image

while True:
    # 1.import img
    success,img=cap.read()
    img=cv.flip(img,1)   # img is mirror type
    # 2 find hand landmark
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        # print(lmList)

        # tip of index and middle finger
        x1,y1=lmList[8][1:]   # first finger 
        x2,y2=lmList[12][1:]  #middle finger


        # 3. check whhich finger r up

        fingers=detector.fingerUp()
        print(fingers)
        # 4. selection mode- 2 finger up
        if fingers[1] and fingers[2]:
            xp,yp=0,0    
            
            print('SELECTION MODE')
            # check for click
            if y1<125:
                if 250<x1<450:
                    header=overlayList[0]
                    drawColor=(255, 255, 255) # white
                elif 550<x1<750:
                    header=overlayList[1]
                    drawColor=(0,0,255) # red
                elif 800<x1<950:
                    header=overlayList[2]
                    drawColor=(255,0,0) # blue
                elif 1050<x1<1200:
                    header=overlayList[3]
                    drawColor=(0,0,0) # black
            cv.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,-1)
                


        # 5. if drawing mode index finger up
        if fingers[1] and fingers[2]==False:
            cv.circle(img,(x1,y1),15,(255,0,255),-1)
            print('DRAWING MODE')
            if xp==0 and yp==0:
                xp,yp=x1,y1
            cv.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
            cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1

    imgGray=cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)    
    _,imgInv=cv.threshold(imgGray,50,255,cv.THRESH_BINARY_INV)
    imgInv=cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img= cv.bitwise_and(img,imgInv)
    img=cv.bitwise_or(img,imgCanvas )

    # set header image
    img[0:125,0:1280]=header     # slicing of img (img[height,width])

    img=cv.addWeighted(img,0.5,imgCanvas,0.5,0)  # overlap 2 imges
    cv.imshow('Image',img)
    cv.imshow('Canvas',imgCanvas)
    cv.imshow('INv',imgInv)
    cv.waitKey(2)     