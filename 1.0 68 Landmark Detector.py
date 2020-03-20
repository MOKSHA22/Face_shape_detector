import cv2
import dlib


camera=cv2.VideoCapture(0)
# (0)  - for the default camera

face_detector=dlib.get_frontal_face_detector()
    # a pre-trained faced detector algorothm in dlib

while(True):
    ret,img=camera.read()

    #cv2.imshow('Live',img)


    #rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #cv2.imshow('RGB',rgb) #BGR2RGB

    #hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cv2.imshow('RGB',hsv) #BGR2HSV

    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # cv2.COLOR_BGR2GRAY - function to convert colour space into gray scale

    rects=face_detector(gray)
        # same like result=algorithm.predict(test_data)
        # as the colour isn't important we need gray images

    for rect in rects:
        # rect - no of faces in the image

        x1=rect.left()
        x2=rect.right()
        y1=rect.top()
        y2=rect.bottom()
            # x1, x2, y1, y2 are the points of the rectagle around the face
        cv2.rectangle(gray,(x1,y1),(x2,y2),(0,255,0),2)
    
    
    cv2.imshow('Gray',gray) # seperate window to show gray scale image called gray

    
    cv2.waitKey(1)
