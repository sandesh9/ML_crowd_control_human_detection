import numpy as np
import math
import cv2
import tkinter as tk
from tkinter import filedialog 
import time  

#input video or use webcam
print("\n\nPlease select an input video ...")
root = tk.Tk()
root.withdraw()
#reference_file_name = filedialog.askopenfilename()
file_name = filedialog.askopenfilename()
cap = cv2.VideoCapture(file_name)

#enable below for webca and disable other file input code
#cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("face.xml")
frn=1
#slope of line
def line1(x,y):
    return y - (9*x)/96.0 - 400

def line2(x,y):
    return y - (9*x)/96.0 - 500

crossedAbove = 0
crossedBelow = 0
points = set()
pointFromAbove = set()
pointFromBelow = set()
H = 1080
W = 1920

#writing video output
OffsetRefLines = 50  #Adjust ths value according to your usage
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_output_test_video.avi',fourcc, 20.0, (W, H))
font = cv2.FONT_HERSHEY_SIMPLEX
#initilizing file output to 0
file1 = open("crossedBelow.txt", "w")  # write mode 
file1.write(str(0)) 
file1.close()
file2 = open("crossedAbove.txt", "w")  # write mode 
file2.write(str(0)) 
file2.close() 
while(1):
    pointInMiddle = set()
    prev = points
    points = set()
    ret, frame1 = cap.read()
    if frame1 is None:
        break
    frame = cv2.resize(frame1,(W, H))
    height = np.size(frame,0)
    width = np.size(frame,1)
    #print(frame)
    #print(height)
    #print(width)
    frame=cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.flip(frame, 1 )
    #oldFgmask = gray.copy()
    
    #input video frame to detect using KNN
    faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
			)
     #write the count of each frame to a file       
    #file1 = open("detect_count_knn5_test.txt", "a")  # write mode 
    #file1.write(str(len(faces)))
    #file1.write("\n")
    #file1.close() 
    for (x, y, w, h) in faces:
        #if 1<w<300 and 1<h<300:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #drawing rectangle in detected area
        point = (int(x+w/2.0), int(y+h/2.0))
        points.add(point)
        #print(faces)
        #print(x)
        #print(y)
        #print(w)
        #print(h)
        #print("Area")
        #print((x+w)*(y+h)+ x*y)
        
           
    for point in points:
        (xnew, ynew) = point
        for prevPoint in prev:
            (xold, yold) = prevPoint
            dist = cv2.sqrt((xnew-xold)*(xnew-xold)+(ynew-yold)*(ynew-yold))
            if dist[0] <= 120:
                if line1(xnew, ynew) >= 0 and line2(xnew, ynew) <= 0:
                    if line1(xold, yold) < 0: # Point entered from line above
                        pointFromAbove.add(point)
                    elif line2(xold, yold) > 0: # Point entered from line below
                        pointFromBelow.add(point)
                    else:   # Point was inside the block
                        if prevPoint in pointFromBelow:
                            pointFromBelow.remove(prevPoint)
                            pointFromBelow.add(point)

                        elif prevPoint in pointFromAbove:
                            pointFromAbove.remove(prevPoint)
                            pointFromAbove.add(point)

                if line1(xnew, ynew) < 0 and prevPoint in pointFromBelow: # Point is above the line
                    print('One Crossed Above')
                    print(point)
                    crossedAbove += 1
                    file1 = open("crossedAbove.txt", "w")  # write mode 
                    file1.write(str(crossedAbove)) 
                    file1.close()
                    pointFromBelow.remove(prevPoint)

                if line2(xnew, ynew) > 0 and prevPoint in pointFromAbove: # Point is below the line
                    print('One Crossed Below')
                    print(point)
                    crossedBelow += 1
                    file1 = open("crossedBelow.txt", "w")  # write mode 
                    file1.write(str(crossedBelow)) 
                    file1.close() 
                    pointFromAbove.remove(prevPoint)
                    
                  
                                       
    #Drawing circle in md point of  detected area                          
    for point in points:
        if point in pointFromBelow:
            cv2.circle(frame, point, 3, (255,0,255),10) #cv2.circle(image, center_coordinates, radius, color, thickness)
        elif point in pointFromAbove:
            cv2.circle(frame, point, 3, (0,255,255),10)
        else:
            cv2.circle(frame, point, 3, (0,0,255),6)
    
    
    cv2.line(frame, (0,400), (1920,580), (255, 0, 0), 2) #cv2.line(image, start_point, end_point, color, thickness)
    cv2.line(frame, (0,500), (1920,680), (255, 0, 0), 2)
    #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    if crossedBelow > 2 and crossedBelow < 5:
        cv2.putText(frame,'Warning!! number of visitor about to exceed',(100,150), font, 1,(0,0,255),2,cv2.LINE_AA)
    elif crossedBelow > 4:
        cv2.putText(frame,'Important!! number of visitor exceeded',(100,200), font, 1,(0,0,255),2,cv2.LINE_AA)
    #cv2.putText(frame,'detected faces = '+str(len(faces)),(100,250), font, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'Travelled Above = '+str(crossedAbove),(100,50), font, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'Travelled Below = '+str(crossedBelow),(100,100), font, 1,(0,0,255),2,cv2.LINE_AA)
    #cv2.putText(frame,'Framenumber = '+str(frn),(100,300), font, 1,(255,0,0),2,cv2.LINE_AA)
    
    #cv2.imshow('a',oldFgmask)
    #frame=cv2.rotate(frame, cv2.ROTATE_180) 
    frn += 1
    cv2.imshow('Out frame',frame)
    #frame = cv.flip(frame, 0)
    frame=cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    out.write(frame)
    l = cv2.waitKey(1) & 0xff
    if l == 27:
        break
cap.release()
cv2.destroyAllWindows()