import cv2 as cv
import time
import numpy as np
import PoseModule as pm

cap=cv.VideoCapture(0)

count=0
dir=0
pTime=0
detector=pm.poseDetector() # create object
while True:

    success,img=cap.read()
    img=cv.resize(img,(1280,720))  #resize video
    # img=cv.imread("AiTrainerProject/img/gy.jpg")
    img=detector.findPose(img,False)
    lmList=detector.findPosition(img,False)
    # print(lmList)
    if len(lmList)!=0:
        # right arm
        # detector.findAngle(img,12,14,16) #landmark

        # left arm
        angle=detector.findAngle(img,11,13,15)
        per=np.interp(angle,(210,320),(0,100))
        bar=np.interp(angle,(210,320),(650,100))
        # print(angle,per)

        # check dumbbell count
        color=(0,255,0)
        if per ==100:
            color=(255,255,0)
            if dir==0:
                count+=0.5
                dir=1

        if per ==0:
            if dir==1:
                count+=0.5
                dir=0
    # draw bar
    cv.rectangle(img,(1100,100),(1175,650),color,3)
    cv.rectangle(img,(1100,int(bar)),(1175,650),color,-1)    
    cv.putText(img,str(int(per)),(1100,75),cv.FONT_HERSHEY_PLAIN,4,(255,0,0),4)

    # draw count
    cv.rectangle(img,(0,450),(250,720),(0,255,0),-1)
    cv.putText(img,str(int(count)),(45,670),cv.FONT_HERSHEY_PLAIN,15,(255,0,0),25)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(img,str(int(fps)),(50,100),cv.FONT_HERSHEY_PLAIN,5,(255,0,0),5)

    print(count)



    cv.imshow("image",img)  
    cv.waitKey(1)