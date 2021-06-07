import cv2
import numpy as np

photo=cv2.imread('r.jpg')
photo1=cv2.imread('bts-pic.jpg')

cv2.imshow('title',photo)
cv2.imshow('title1',photo1)
cv2.waitKey()    #argument in milliseconds
cv2.destroyAllWindows()

#to detect faces in both images
model=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#to detect face and get coordinates from photo1
face1=model.detectMultiScale(photo1)
x11=face1[0][0]
y11=face1[0][1]
x21= x11+face1[0][2]
y21= y11+face1[0][3]

#to print photo with green rectangle around detected face
aphoto=cv2.rectangle(photo1,(x11,y11),(x21,y21),[0,255,0],2)
cv2.imshow('bts',aphoto)
cv2.imwrite('bts-detect.jpg',aphoto)
cv2.waitKey()    
cv2.destroyAllWindows()

#to detect face and get coordinates from photo
face=model.detectMultiScale(photo)
x1=face[0][0]
y1=face[0][1]
x2= x1+face[0][2]
y2= y1+face[0][3]

#to print photo with green rectangle around detected face
bphoto=cv2.rectangle(photo,(x1,y1),(x2,y2),[0,255,0],2)
cv2.imshow('fb',bphoto)
cv2.imwrite('fb-detect.jpg',aphoto)
cv2.waitKey()    
cv2.destroyAllWindows()


#to crop from each picture
photo=cv2.imread('r.jpg')
photo1=cv2.imread('bts-pic.jpg')

crop_photo=photo[y1:y2,x1:x2]
cv2.imshow('title',crop_photo)
cv2.imwrite("fbcrop.jpg",crop_photo)
cv2.waitKey()    
cv2.destroyAllWindows()

crop_photo1=photo1[y11:y21,x11:x21]
cv2.imshow('title',crop_photo1)
cv2.imwrite("btscrop.jpg",crop_photo1)
cv2.waitKey()    
cv2.destroyAllWindows()

#to swap both pictures
photo=cv2.imread('r.jpg')
photo1=cv2.imread('bts-pic.jpg')

x_end1=x11+crop_photo.shape[1]
y_end1=y11+crop_photo.shape[0]
photo1[y11:y_end1,x11:x_end1]=crop_photo

cv2.imshow('btsswap',photo1)
cv2.imwrite("btsswap.jpg",photo1)
cv2.waitKey()    
cv2.destroyAllWindows()

photo=cv2.imread('r.jpg')
photo1=cv2.imread('bts-pic.jpg')

x_end=x1+crop_photo1.shape[1]
y_end=y1+crop_photo1.shape[0]
photo[y1:y_end,x1:x_end]=crop_photo1

cv2.imshow('fbswap',photo)
cv2.imwrite("fbswap.jpg",photo)
cv2.waitKey()    
cv2.destroyAllWindows()
