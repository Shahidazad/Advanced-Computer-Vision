# we create module for reuse of code


import cv2  as cv
import mediapipe as mp # this framwork use RGB
import time 
import math



class poseDetector():

    def __init__(self,mode=False,upBody=False,smooth=True,detectionCon=True,trackCon=0.5):

        self.mode=mode
        self.upBody=upBody
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectionCon,self.trackCon)

    def findPose(self,img,draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)  #framwork use RGB thats y we cvt it
        self.results=self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)# connect with lines

        return img




    def findPosition(self,img,draw=True):
        self.lmList=[]
        if self.results.pose_landmarks:

            for id,lm   in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                # print(id,lm)
                cx,cy=int(lm.x*w),int(lm.y*h)         # exact pixel value            cv.circle(img,(cx,cy),10,(255,0,0),-1)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),10,(255,0,0),-1)
        return self.lmList

    def findAngle(self,img,p1,p2,p3,draw=True):  # find angle
        
        x1,y1=self.lmList[p1][1:]
        x2,y2=self.lmList[p2][1:]
        x3,y3=self.lmList[p3][1:]


        # calculate angle
        angle=math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
        
        if angle<0: # for -ve value
            angle+=360
        # print(angle)


        if draw:
            
            cv.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv.line(img,(x2,y2),(x3,y3),(255,255,255),3)
            cv.circle(img,(x1,y1),15,(255,0,0),-1)
            cv.circle(img,(x1,y1),5,(255,0,0),2)
            cv.circle(img,(x2,y2),15,(255,0,0),-1)
            cv.circle(img,(x1,y1),5,(255,0,0),2) 
            cv.circle(img,(x3,y3),15,(255,0,0),-1)
            cv.circle(img,(x1,y1),5,(255,0,0),2)
            # cv.putText(img,str(int(angle)),(x2-80,y2+10),cv.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
        return angle


def main():
    cap=cv.VideoCapture('pose estimation project/videos/v.mp4')
    pTime=0
    mpDraw=mp.solutions.drawing_utils
    detector=poseDetector() 
    while True:
        success,img=cap.read()
        img=detector.findPose(img)
        lmList=detector.findPosition(img)

        cv.imshow('img',img)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

    

    cv.putText(img,str(int(fps)),(70,50),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv.imshow('img',img)
    cv.waitKey(2)


if __name__=="__main__":
    main()