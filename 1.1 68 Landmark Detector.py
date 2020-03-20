import cv2
import dlib
from imutils import face_utils

camera=cv2.VideoCapture(0)
# (0)  - for the default camera

face_detector=dlib.get_frontal_face_detector()
    # a pre-trained faced detector algorothm in dlib

landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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
            # [x1, x2, y1, y2] are the points of the rectagle around the face
            # stores as list elements for each face detected
            # {[x1, x2, y1, y2],[x1, x2, y1, y2]} when there are two faces in the image

        cv2.rectangle(gray,(x1,y1),(x2,y2),(255,0,0),2)
            # draw a rectangle around the face

        points=landmark_detector(gray,rect)
            # passing thr gray image with ROI, returns the 68 points
            # points are not numpy array
            # to covert this into numpy array use imutils - image utilities library

        points=face_utils.shape_to_np(points)
            # converting the points object into a numpy array

        for point in points: # runs for 68 times

            xp=point[0]
            yp=point[1]
            # two dimensional array with 68 raws and 2 columns for x and y coordinate of the each point

            cv2.circle(gray,(xp,yp),2,(255,0,255),-1)
            # printing points
            


    cv2.imshow('Gray',gray) # seperate window to show gray scale image called gray

    
    cv2.waitKey(1)
