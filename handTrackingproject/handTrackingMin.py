
import cv2 as cv
import mediapipe as mp
import time

cap=cv.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

pTime=0 # previous time
cTime=0 # current time
while True:
    # w=1
    success,img=cap.read()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:# tracking each hand
            for id,lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c=img.shape # height width channel of img
                cx,cy=int(lm.x*w),int(lm.y*h)# position 
                print(id,cx,cy)  # print id also for location for that id
                # if id==0:  # draw circle at landmark first
                cv.circle(img,(cx,cy),15,(255,0,255),cv.FILLED)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    # w=input("enter")
    # if w==0:
    #     break

    

    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv.imshow('Image',img)
    cv.waitKey(1)
    