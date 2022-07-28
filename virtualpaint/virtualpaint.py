import cv2 as cv
import time  # show the frame rate
import os    # to access the images
import numpy as np
import handTrackingModule as htm 

folderPath='virtualpaint/img'
myList=os.listdir(folderPath)
print(myList)


overlayList=[]
for imPath in myList:
    image=cv.imread(f'{folderPath}/{imPath}') # for loop is appied on 4 images
    overlayList.append(image)
print(len(overlayList))
header=overlayList[0]
drawColor=(255,0,255) #initial value
cap=cv.VideoCapture(0)   # if u have multiple camera then 1 to run the web camera
cap.set(3,1280)  # width should be same  size of web cam
cap.set(4,720)   # height should be same size of web cam
detector=htm.handDetector()  # it is a hand tracking detector 
brushThickness=15
eraserThichness = 50

xp,yp=0,0 #initial value

imgCanvas=np.zeros((720,1280,3),np.uint8)  # new image,we draw on imgCanvas

while True:
    # 1.import img
    success,img=cap.read()
    img=cv.flip(img,1)   # img is mirror type (horizontal flip)


    # 2 find hand landmark   (using hand tracking module)
    img=detector.findHands(img)   # hand will be track
    lmList=detector.findPosition(img,draw=False)  # Landmarks of hand
    if len(lmList)!=0:
        # print(lmList)

        # tip of index and middle finger
        x1,y1=lmList[8][1:]   # first finger [8,x,y] we want x,y
        x2,y2=lmList[12][1:]  # middle finger


        # 3. check which fingers are up

        fingers=detector.fingerUp()
        print(fingers)
        # 4. selection mode  2 finger up
        if fingers[1] and fingers[2]:
            xp,yp=0,0    
            
            print('SELECTION MODE')
            # check for click on frame
            if y1<125:   # 125 is size of header
                if 250<x1<450:   # location of first paint
                    header=overlayList[0]
                    drawColor=(250, 235, 235) # white
                elif 550<x1<750:   # location of second  paint
                    header=overlayList[1]
                    drawColor=(0,0,255) # red
                elif 800<x1<950:     # location of third paint
                    header=overlayList[2]
                    drawColor=(255,0,0) # blue
                elif 1050<x1<1200:   # location of rubber paint
                    header=overlayList[3]
                    drawColor=(0,0,0) # black
            cv.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,-1) 
                


        # 5. if drawing mode index finger is up (1 finger)
        if fingers[1] and fingers[2]==False:  # second fingers should be down
            cv.circle(img,(x1,y1),15,drawColor,-1)  # circle on tip of first finger
            print('DRAWING MODE')
            if xp==0 and yp==0: # our line will not  start from origin 
                xp,yp=x1,y1     # current position

            if drawColor == (0,0,0):
                cv.line(img,(xp,yp),(x1,y1),drawColor,eraserThichness)
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThichness)

            else:
                cv.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1

    imgGray=cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)   # covt to gray scale img 
    _,imgInv=cv.threshold(imgGray,50,255,cv.THRESH_BINARY_INV)  # convt black to white ,color to black
    imgInv=cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR) # convt  to gray scale to color
    img= cv.bitwise_and(img,imgInv)
    img=cv.bitwise_or(img,imgCanvas )

    # set header image
    img[0:125,0:1280]=header     # slicing of img (img[height,width])

    img=cv.addWeighted(img,0.5,imgCanvas,0.5,0)  # overlap 2 images now we draw on original img
    cv.imshow('Image',img)
    cv.imshow('Canvas',imgCanvas)
    cv.imshow('INv',imgInv)
    cv.waitKey(2)     