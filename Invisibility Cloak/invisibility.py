import numpy as np
import cv2

def nothing(x):
    pass

cap=cv2.VideoCapture(0)

#capturing instances
for i in range(100):
    rd,background=cap.read()
background=np.flip(background,axis=1)

cv2.namedWindow('HSV palette')
cv2.createTrackbar('LH','HSV palette',0,179,nothing)
cv2.createTrackbar('LS','HSV palette',0,255,nothing)
cv2.createTrackbar('LV','HSV palette',0,255,nothing)

cv2.createTrackbar('UH','HSV palette',0,179,nothing)
cv2.createTrackbar('US','HSV palette',0,255,nothing)
cv2.createTrackbar('UV','HSV palette',0,255,nothing)




kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))



while True:
    _,frame= cap.read()
    frame=np.flip(frame,axis=1)

    lh = cv2.getTrackbarPos('LH','HSV palette')
    ls = cv2.getTrackbarPos('LS', 'HSV palette')
    lv = cv2.getTrackbarPos('LV', 'HSV palette')

    uh = cv2.getTrackbarPos('UH', 'HSV palette')
    us = cv2.getTrackbarPos('US', 'HSV palette')
    uv = cv2.getTrackbarPos('UV', 'HSV palette')

    '''lower_blue = np.array([lh, ls, lv])
    upper_blue = np.array([uh, us, uv])'''

    lower_blue = np.array([59,109,0])
    upper_blue = np.array([179,255,255])



    framehsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    framehsv=cv2.GaussianBlur(framehsv,(5,5),0)
    mask=cv2.inRange(framehsv,lower_blue,upper_blue)
    cv2.imshow('masked',mask)

    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernel1)

    inv_mask=cv2.bitwise_not(mask)

    no_cloak=cv2.bitwise_and(frame,frame,mask=inv_mask)
    cloak=cv2.bitwise_and(background,background,mask=mask)

    output=cv2.addWeighted(no_cloak,1,cloak,1,0)

    cv2.imshow('invisibility',output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
