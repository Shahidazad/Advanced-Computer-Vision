# we create module for reuse of code


import cv2  as cv
import mediapipe as mp # this framwork use RGB
import time 



class poseDetector():

    def __init__(self,mode=False,upBody=False,smooth=True,detectionCon=0.5,trackCon=0.5):

        self.mode=mode
        self.upBody=upBody
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.smooth=smooth
        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.upBody,self.detectionCon,self.trackCon,self.smooth)

    def findPose(self,img,draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)  #framwork use RGB thats y we cvt it
        self.results=self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)# connect with lines

        return img




    def findPosition(self,img,draw=True):
        lmList=[]
        if self.results.pose_landmarks:

            for id,lm   in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                # print(id,lm)
                cx,cy=int(lm.x*w),int(lm.y*h)         # exact pixel value            cv.circle(img,(cx,cy),10,(255,0,0),-1)
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),10,(255,0,0),-1)
        return lmList



    


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