import numpy as np
import math
import cv2 as cv
from utils import image_resize

frontalface_location = 'haarcascade_frontalface_default.xml'
polarbear_location = "polarbear.png"
face_cascade = cv.CascadeClassifier(frontalface_location)
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 80)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 60)
cap.set(cv.CAP_PROP_FPS, 24)
polarbear = cv.imread(polarbear_location, -1)

while (True):
    ret, frame = cap.read()

    bear = cv.cvtColor(polarbear.copy(), cv.COLOR_BGR2BGRA)
    frame_h, frame_w, frame_c = frame.shape
    overlay = np.zeros((frame_h, frame_w, 4), dtype = 'uint8') #sets every pixel value to 0
    frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor = 1.1, 
            minNeighbors = 5,
            minSize = (1, 1),
            flags = cv.CASCADE_SCALE_IMAGE
        )

    for (x,y,w,h) in faces:
        bear = image_resize(bear.copy(), height = h)
        pb_h, pb_w, pb_c = bear.shape
        width_mult = int(math.floor(0.4 * w))
        height_mult = int(math.floor(0.4 * h))
        

        for i in range(0, pb_h):
            for j in range(0, pb_w):
                if bear[i, j][3] != 0:
                    overlay[y + i - height_mult, x + j - width_mult] = bear[i, j]

        cv.addWeighted(overlay, 1.0, frame, 1.0, 0, frame)
        #cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
    
    cv.imshow('Polarbear Detection',frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# housekeeping
cap.release()
cv.destroyAllWindows()

