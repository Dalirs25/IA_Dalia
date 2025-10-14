
import cv2 as cv
import numpy as np


vCap = cv.VideoCapture(0);
faceDetector = cv.CascadeClassifier('../haarcascade_frontalface_alt.xml')

x=y=w=h=0
img: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)
count = 0

imageIndex = 916
while True:
    ret, frame = vCap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # cvtColor, scaleFacto, neightboursCount
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces: 
        m = int(h/2)

        # Draw rectangles for the detected faces
        # frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # frame = cv.rectangle(frame, (x, y+m), (x + w, y + h), (0, 255, 0), 2)

        faceDetectedFrame = frame[y: y + h, x: x + w]
        faceDetectedFrame = cv.resize(faceDetectedFrame, (100, 100), interpolation=cv.INTER_AREA)

        if (imageIndex % 1 == 0):
            cv.imwrite('C:/python_projects/face_recog/rostros/datasets/dalia/dalia'+str(imageIndex)+'.jpg', faceDetectedFrame)
            cv.imshow('FACE DETECTED FRAME', faceDetectedFrame)


        roi = frame[y:y + h, x:x + w]
        img = cv.subtract(np.full_like(roi, 180), roi)
        count += 1

    cv.imshow('ROSTROS', frame)
    # cv.imshow('CARA', img)
    imageIndex += 1
    k = cv.waitKey(1)
    if k == 27:
        break


vCap.release()
cv.destroyAllWindows()
