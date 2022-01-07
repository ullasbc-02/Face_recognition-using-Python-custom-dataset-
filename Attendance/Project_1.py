import cv2 as cv
import numpy as np
import face_recognition as fr #It uses HOG method to detect face (dlib library helps in plotting)
import os
from datetime import datetime

path = 'F:\PycharmProjects\Face Recognition\Images_Attendance' #includes images of modi,rahul gandhi,kejriwal
images = [] #Includes images
classNames = [] #Includes file names
myList = os.listdir(path)
# print(myList)

for cl in myList:
 curImg = cv.imread(f'{path}/{cl}')
 images.append(curImg)
 classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
  encodeList = []
  for img in images:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    encode = fr.face_encodings(img)[0]
    encodeList.append(encode)
  return encodeList

encodeListKnown = findEncodings(images)
print("Encoding completed")

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
         entry = line.split(',')
         nameList.append(entry[0])
        if name not in nameList:
         now = datetime.now()
         dtString = now.strftime('%H:%M:%S')
         f.writelines(f'\n{name},{dtString}')


cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    facesCurFrame = fr.face_locations(imgS) #We may find multiple faces
    encodesCurFrame = fr.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): #want in same loop so zip is used
      matches = fr.compare_faces(encodeListKnown, encodeFace)
      faceDis = fr.face_distance(encodeListKnown, encodeFace)
      print(faceDis)
      matchIndex = np.argmin(faceDis)
      if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
        cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        markAttendance(name)

    cv.imshow('Webcam', img)
    cv.waitKey(1)