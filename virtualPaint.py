import cv2
import numpy as np
import os
import time
import hand_tracking_module as htm

# Canvas size
wCam, hCam = 1280, 680

# Camera setup
video = cv2.VideoCapture(0)
video.set(3, wCam)
video.set(4, hCam)

# Load header images
folderPath = "paint"
overlayList = [cv2.imread(os.path.join(folderPath, img)) for img in os.listdir(folderPath)]
header = overlayList[0]

# Drawing settings
drawColor = (255, 255, 0)
brushThickness = 5
eraserThickness = 25

# Hand detector
detector = htm.handDetector()
xp, yp = 0, 0
imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)

while True:
    # Capture frame
    success, frame = video.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (wCam, hCam))

    # Find hand and landmarks
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    if lmList:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingerUp(lmList)

        # Selection mode
        if fingers[1] and fingers[2]:
            if y1 < 105:
                if 183 < x1 < 365: drawColor = (0, 255, 0)
                elif 366 < x1 < 548: drawColor = (255, 0, 255)
                elif 549 < x1 < 731: drawColor = (255, 0, 0)
                elif 732 < x1 < 914: drawColor = (0, 0, 255)
                elif 915 < x1 < 1097: drawColor = (255, 255, 255)
                elif 1098 < x1 < 1279: drawColor = (0, 0, 0)

            cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Drawing mode
        # Drawing mode
        if fingers[1] and not fingers[2]:
            cv2.circle(frame, (x1, y1), 10, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
            cv2.line(frame, (xp, yp), (x1, y1), drawColor, thickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
            xp, yp = x1, y1
        else:
            xp, yp = 0, 0


    # Canvas blending
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    imgInv = cv2.resize(imgInv, (wCam, hCam))

    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, imgCanvas)

    # Overlay header
    header_resized = cv2.resize(header, (wCam, 105))
    frame[0:105, 0:wCam] = header_resized

    # Show windows
    cv2.imshow('Painter', frame)
    # cv2.imshow('Canvas', imgCanvas)

    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

video.release()
cv2.destroyAllWindows()
