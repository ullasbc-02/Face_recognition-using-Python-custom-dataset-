import cv2 as cv
import pandas as pd
import numpy as np
import face_recognition as fr

image1 = cv.imread('tata.png')
image2 = cv.imread('tata2.png')


faceLoc1 = fr.face_locations(image1)[0]
encodetata1 = fr.face_encodings(image1)[0]
cv.rectangle(image1,(faceLoc1[3],faceLoc1[0]),(faceLoc1[1],faceLoc1[2]),(255,255,255),2)
# print(faceLoc1)

faceLoc2 = fr.face_locations(image2)[0]
encodetata2 = fr.face_encodings(image2)[0]
cv.rectangle(image2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(255,255,255),2)

cmp = fr.compare_faces([encodetata1],encodetata2)
print(cmp)
dis = fr.face_distance([encodetata1],encodetata2)
print(dis)
cv.putText(image2,f'{cmp} {round(dis[0],2)}',(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)


cv.imshow('Tata',image2)
cv.waitKey(0)