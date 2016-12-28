import cv2
import sys

import numpy as np

classifier = sys.argv[1]
faceDetect = cv2.CascadeClassifier(classifier)
cam = cv2.VideoCapture(0)


id = raw_input('enter user id  ')
sampleNum = 0
while (True):
      ret, img = cam.read()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      faces = faceDetect.detectMultiScale( 
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (25, 25),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
       )

      for(x,y,w,h) in faces:
           sampleNum = sampleNum+1
           cv2.imwrite("dataset/User."+str(id)+"."+str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
           cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
           cv2.waitKey(100)
      cv2.imshow("Face", img)
      cv2.waitKey(1)

      if(sampleNum>50):   
           break

cam.release()
cv2.destroyAllWindows()

