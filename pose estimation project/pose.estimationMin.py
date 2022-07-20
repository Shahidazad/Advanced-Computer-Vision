
import cv2  as cv
import mediapipe as mp # this framwork use RGB
import time 
mpPose=mp.solutions.pose
pose=mpPose.Pose()
cap=cv.VideoCapture('pose estimation project/videos/v.mp4')
pTime=0
mpDraw=mp.solutions.drawing_utils 
while True:
    succes,img=cap.read() # this img is BGR
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)  #framwork use RGB thats y we cvt it
    results=pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)# connect with lines 
        for id,lm   in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            print(id,lm)
            cx,cy=int(lm.x*w),int(lm.y*h)         # exact pixel value
            cv.circle(img,(cx,cy),10,(255,0,0),-1)




    cv.imshow('img',img)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    

    cv.putText(img,str(int(fps)),(70,50),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv.imshow('img',img)
    cv.waitKey(2)