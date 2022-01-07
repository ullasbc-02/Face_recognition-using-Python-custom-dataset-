import cv2 as cv
import pandas as pd
import numpy as np

import face_recognition as fr

image = cv.imread('bigbang.png')
locs = fr.face_locations(image)
print(len(locs), " faces detected")

for face in locs:
  cv.rectangle(image, (face[3], face[0]), (face[1], face[2]), (255, 255, 255), 1)
  cv.imshow('Detected',image)
cv.waitKey(0)
# cv.destroyWindow()
