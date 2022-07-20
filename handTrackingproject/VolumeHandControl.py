import cv2 as cv
import numpy as np
import time
import handTrackingModule as htm        # reduce redundancy
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  #control volume of computer

wCam,hCam=640,480   # size of web cam

cap=cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

pTime=0

detector=htm.handDetector()  # create object


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()

volRange=volume.GetVolumeRange()

minVol=volRange[0]
maxVol=volRange[1]
vol=0
volper=0
volbar=300
while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        # print(lmList[4],lmList[8])  # pts of thumb and finder first
        x1,y1=lmList[4][1],lmList[4][2]#pts of thumb
        x2,y2=lmList[8][1],lmList[8][2]#pts of  first finder
        cx,cy=(x1+x2)//2,(y1+y2)//2

        cv.circle(img,(x1,y1),15,(255,0,255),-1) #draw circle
        cv.circle(img,(x2,y2),15,(255,0,255),-1) #draw circle
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),3)# draw line        
        cv.circle(img,(cx,cy),15,(255,0,255),-1)
        length=math.hypot(x2-x1,y2-y1)
        # print(length)

        # hand range 50-300
        # volume range -65-0

        vol=np.interp(length,[10,206],[minVol,maxVol])  # converting hand range to vol range
        volume.SetMasterVolumeLevel(vol, None)
        volper=np.interp(length,[10,206],[0,100])
        volbar=np.interp(length,[10,206],[300,150])
        print(int(length),vol)


        if length<29:
            cv.circle(img,(cx,cy),17,(0,255,0),-1)

    cv.rectangle(img,(50,150),(85,300),(0,255,0),2)
    cv.rectangle(img,(50,int(volbar)),(85,300),(0,255,0),-1)
    cv.putText(img,f'per:{int(volper)}%',(50,350),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv.putText(img,f'FPS:{int(fps)}',(40,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv.imshow('IMG',img)
    cv.waitKey(1)