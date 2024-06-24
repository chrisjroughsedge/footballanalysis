import cv2 
  
  
def mouse_click(event, x, y, flags, param): 
    # to check if left mouse button was clicked 
    if event == cv2.EVENT_LBUTTONDOWN: 
        font = cv2.FONT_HERSHEY_SIMPLEX
        print(f'[{x},{y}]')
        cv2.putText(frame, str(x) + ',' + str(y), (x,y), font, 1, (255,0,0), 2)
        
        cv2.imwrite("frame.jpg", param) 
  
    # to check if right mouse button was clicked 
    if event == cv2.EVENT_RBUTTONDOWN: 
        print("right click") 
        cv2.imshow("Current Frame", frame) 
  
  
cap = cv2.VideoCapture("../input_videos/ShortVid.mp4") 
  
if cap.isOpened() == False: 
    # give error message 
    print("Error in opening file.") 
else: 
    # proceed forward 
    while(cap.isOpened()): 
        ret, frame = cap.read() 
        if ret == True: 
            cv2.imshow("GFG", frame)
            cv2.setMouseCallback('GFG', mouse_click, param=frame)
            # if cv2.waitKey(11) & 0xFF == ord('f'):
            #     print(11)
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
        else: 
            break
  
  
cap.release() 
cv2.destroyAllWindows() 