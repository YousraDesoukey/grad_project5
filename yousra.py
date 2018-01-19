import numpy as np
import cv2

#read image
img = cv2.imread('u1.png')

#resizing
img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)

#greyscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

#thresholding
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('Thresholding', thresh)

# noise removal
#opening
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
cv2.imshow('Opening', opening)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=2)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
prep = cv2.subtract(sure_bg,sure_fg)


cv2.imshow('Preprocessed', prep)

#find contours
image,contours,hierarchy = cv2.findContours(prep,2,1)

#draw contours
img2 = cv2.drawContours(image , contours, -1, (150,255,0), 3)

cv2.imshow('All Contours',img2)

#sort contours
cnts = sorted(contours, key = cv2.contourArea, reverse=True)[:2]

#loop on contours if more than 1
for cnt in cnts:
    if ((cv2.contourArea(cnt) < cv2.contourArea(cnts[0])-cv2.contourArea(cnts[0])*0.2)):
        break
    else:
        x,y,w,h = cv2.boundingRect(cnt)
        final = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),3)
cv2.imshow('Final Image',final)
cv2.waitKey(0)