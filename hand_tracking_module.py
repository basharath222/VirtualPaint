import cv2 as cv
import mediapipe as mp
import numpy as np
import time


class handDetector():
    def __init__(self,mode = False, maxHands = 2,detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.tipIds = [4,8,12,16,20]

        self.mpdraw = mp.solutions.drawing_utils

    def findHands(self,frame,draw = True):   
        rgb_img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.result = self.hands.process(rgb_img)

        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(frame, handlms, self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findPosition(self, frame, handNo = 0,draw = True):
        lmlist = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])
                
                if draw:
                    cv.circle(frame,(cx,cy),5,(0,255,0),5)
        
        return lmlist

    def fingerUp(self,lmlist):
        
        fingers = []
        if lmlist[self.tipIds[0]][1] < lmlist[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        

        for id in range(1,5):

            if lmlist[self.tipIds[id]][2] < lmlist[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers            



# video.release()
# cv.destroyAllWindows()

def main():
    cTime  = 0
    pTime = 0
    video = cv.VideoCapture(0)
    detector  = handDetector()
    while True:
        isTrue, frame = video.read()
        frame = detector.findHands(frame)
        lmlist = detector.findPosition(frame)
        if len(lmlist) > 4:
            print(lmlist[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv.putText(frame,str(int(fps)),(20,50),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

        cv.imshow('video',frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break


if __name__ == "__main__":
    main()