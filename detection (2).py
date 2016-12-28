import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cam = cv2.VideoCapture(0)
rec = cv2.createLBPHFaceRecognizer()
rec.load("identification//training_data.yml")
id = 0
names = ['Cesar', 'Cassiano', 'J', 'Sang', 'Cooper', 'Andre'] # names for the tags or ids


font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,2,1,0,4)
while True:
      ret, frame = cam.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      faces = faceDetect.detectMultiScale( 
        gray,
        scaleFactor = 1.5,
        minNeighbors = 5,
        minSize = (25, 25),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
       )

      for(x,y,w,h) in faces:
           cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 125), 2)
           roi_gray = gray[y:y + h, x:x+w]
           roi_color = frame[y:y + h, x:x+w]
           eyes = eye_cascade.detectMultiScale(roi_gray)          
           for (ex, ey, ew, eh) in eyes:
                 cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
           id, conf = rec.predict(gray[y:y + h, x: x + w])
           if (conf < 100):
              id = names[id - 1]
           else:
               id = "unknown", 
           confidence = conf
           cv2.cv.PutText(cv2.cv.fromarray(frame), str(confidence), (x  - h, y), font, 80)    
           cv2.cv.PutText(cv2.cv.fromarray(frame), str(id), (x, y + h), font, 255)
           print (id, conf)
           
      cv2.imshow('Video', frame)
      if(cv2.waitKey(1) == ord('q')):   
           break

cam.release()
cv2.destroyAllWindows()

