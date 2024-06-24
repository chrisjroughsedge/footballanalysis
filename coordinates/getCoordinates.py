import cv2
import numpy as np

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'[{x},{y}]')
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 1, (255,0,0), 2)
        cv2.imshow('image', img)
    
    if event == cv2.EVENT_RBUTTONDOWN:
       print(f'[{x},{y}]')
       b = img[y, x, 0]
       g = img[y, x, 1]
       r = img[y, x, 2] 
       cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x,y), font, 1, (255,255,0), 2)
       cv2.imshow('image', img)


def click_event2(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'[{x},{y}]')
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img2, str(x) + ',' + str(y), (x,y), font, 1, (255,0,0), 2)
        cv2.imshow('image2', img2)
    
    if event == cv2.EVENT_RBUTTONDOWN:
       print(f'[{x},{y}]')
       b = img2[y, x, 0]
       g = img2[y, x, 1]
       r = img2[y, x, 2] 
       cv2.putText(img2, str(b) + ',' + str(g) + ',' + str(r), (x,y), font, 1, (255,255,0), 2)
       cv2.imshow('image2', img2)

if __name__=="__main__":
    img = cv2.imread('images/Right.png')
    cv2.imshow('image', img)
    img2 = cv2.imread('images/statsbombpitch.png')
    cv2.imshow('image2', img2)
    cv2.setMouseCallback('image', click_event)
    cv2.setMouseCallback('image2', click_event2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
