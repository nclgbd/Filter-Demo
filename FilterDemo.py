import numpy as np
import math
import cv2 as cv

LOGITECH_CAMERA_PORT = 1
frontalface_location = 'haarcascades/haarcascade_frontalface_default.xml'
img_path = 'media/dog_filter_trans.png'
face_cascade = cv.CascadeClassifier(frontalface_location)
cap = cv.VideoCapture(LOGITECH_CAMERA_PORT)
cap.set(cv.CAP_PROP_FPS, 60)
filter_img = cv.imread(img_path, -1)

cv.namedWindow("WiCS Demo", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("WiCS Demo",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)


while (True):
    ret, frame = cap.read()

    c_filter_img = cv.cvtColor(filter_img.copy(), cv.COLOR_BGR2BGRA)
    frame_h, frame_w, frame_c = frame.shape
    overlay = np.zeros((frame_h, frame_w, 4), dtype = 'uint8') #sets every pixel value to 0
    frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor = 1.4, 
            minNeighbors = 2,
            minSize=(30,30),
            flags = cv.CASCADE_SCALE_IMAGE
        )

    '''
    for (x,y,w,h) in faces:
        bear = cv.resize(bear, (w, h))
        pb_h, pb_w, pb_c = bear.shape
        
        for i in range(pb_h):
            for j in range(pb_w):
                if bear[i,j][3] != 0:
                    overlay[i,j] = bear[i,j]
                    
        cv.addWeighted(overlay, 1.0, frame, 1.0, 0, frame)
    '''
    
    for (x,y,w,h) in faces:
        # print(x)
        w = int(1.5 * w)
        h = int(1.75 * h)
        y = y - int(0.375 * h)
        x = x - int(0.15 * w)
        c_filter_img = cv.resize(c_filter_img, (w, h))
        if c_filter_img.shape == overlay[y:y+h, x:x+w].shape:
            overlay[y:y+h, x:x+w] = c_filter_img
            
    cv.addWeighted(overlay, 1.0, frame, 1.0, 0, dst=frame)
     
    
    
    cv.imshow("WiCS Demo", frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# housekeeping
cap.release()
cv.destroyAllWindows()

