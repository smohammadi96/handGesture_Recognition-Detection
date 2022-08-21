from collections import deque
import numpy as np
import cv2
import pyautogui as pt
def direction():
        pt.FAILSAFE = False
        handCascade = cv2.CascadeClassifier("hand.xml")
       
        DEQUE_MAX_LEN =32
        pts = deque(maxlen=DEQUE_MAX_LEN)
        counter = 0
        (dX, dY) = (0, 0)
        direction = ""
        camera = cv2.VideoCapture(0)
        SPEED = 20
        
   
        while True:
            ret, frame = camera.read()
            frame = cv2.flip(frame,1)
            flag  = False
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            hands = handCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
            for (x, y, w, h) in hands:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center = (x+w//2, y+h//2)
        
                flag = True
        
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)
       
            for i in np.arange(1, len(pts)):
            
            
                if pts[i - 1] is None or pts[i] is None:
                    continue
       
        
                if counter >= 10 and i == 1 and pts[-1] is not None:
             
                    dX = pts[-1][0] - pts[i][0]
                    dY = pts[-1][1] - pts[i][1]
                    (dirX, dirY) = ("", "")
              
                    if np.abs(dX) > 10:
                        dirX = "Left" if np.sign(dX) == 1 else "Right"
        
                    if np.abs(dY) > 10:
                        dirY = "Up" if np.sign(dY) == 1 else "Down"
        
                    if dirX != "" and dirY != "":
                        direction = "{}-{}".format(dirY, dirX)
        
                  
                    else:
                        direction = dirX if dirX != "" else dirY
                    if flag:
                        if dX > 0:
                            s = -SPEED
                        else:
                            s = SPEED
                        pt.moveRel(s, 0)
        
                        if dY > 0:
                            s = -SPEED
                        else:
                            s = SPEED
                        pt.moveRel(0, s)
       
                thickness = int(np.sqrt(DEQUE_MAX_LEN / float(i + 1)) * 1.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        
           
            cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 255), 3)
            cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)
        
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            counter += 1
           
            if key == ord("q"):
                break  
        camera.release()
        cv2.destroyAllWindows()
