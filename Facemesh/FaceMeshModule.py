import cv2 as cv

import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        
        self.maxFaces=max_num_faces
        self.minDetectionCon=min_detection_confidence
        self.refine_landmarks=refine_landmarks
        self.minTrackCon=min_tracking_confidence

        self.mpDraw=mp.solutions.drawing_utils  # drawing pt on face
        self.mpFaceMesh=mp.solutions.face_mesh
        self.faceMesh=self.mpFaceMesh.FaceMesh(max_num_faces=3,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1,circle_radius=2)

    def findFaceMesh(self,img,draw=True):  # 478 pts on face
        self.imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB) # grayscale
        results=self.faceMesh.process(self.imgRGB)
        faces=[]
        if results.multi_face_landmarks:
            
            for faceLms in results.multi_face_landmarks: # landmark of one pt (faceLms)
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_TESSELATION,self.drawSpec,self.drawSpec)
                face=[]
                for id,lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih,iw,ic=img.shape
                    x,y=int(lm.x*iw),int(lm.y*ih) # convert values in pixels
                    cv.putText(img,str(id),(x,y),cv.FONT_HERSHEY_PLAIN,1,(0,255,0),1)  # 1=scale,1=thickness
                    # print(id,x,y)   # x,y are pts on face 
                    face.append([x,y])
                faces.append(face)
        return img,faces

    


def main():
    cap=cv.VideoCapture(0)
    pTime=0
    detector=FaceMeshDetector()

    while True:
        success,img=cap.read()
        img,faces=detector.findFaceMesh(img)
        if len(faces)!=0:
            print(len(faces[0]))

        cTime=time.time()  # frame rate
        fps=1/(cTime-pTime)
        pTime=cTime
        cv.putText(img,f'FPS:{int(fps)}',(10,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv.imshow('Image',img)
        cv.waitKey(1)


if __name__=="__main__":
    main()