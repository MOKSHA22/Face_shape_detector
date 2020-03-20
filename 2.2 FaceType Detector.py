import os
import cv2
import dlib
from imutils import face_utils


camera=cv2.VideoCapture(0)


face_detector=dlib.get_frontal_face_detector()
    # a pre-trained faced detector algorothm in dlib

landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # a pre-trained external shape_predictor_68_face_landmarks


import joblib

algorithm=joblib.load('KNN_model.sav')
    # loading the pre-trained algithm that we saved


datapath='Face Shapes'
    # data path is given, folder name


labels=os.listdir(datapath)
    # os.listdir(datapath) - returns directories in the given path

label_dict={}

for i in range(len(labels)):
    label_dict[i]=labels[i]

print(label_dict)

def predict_face_type(img): # function to  predict face type
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # cv2.COLOR_BGR2GRAY - function to convert colour space into gray scale

    rects=face_detector(gray)
        # same like result=algorithm.predict(test_data)
        # as the colour isn't important we need gray images



    for rect in rects:
        # rect - no of faces in the image
        # runs as the number of faces detected

        x1=rect.left()
        x2=rect.right()
        y1=rect.top()
        y2=rect.bottom()
        # ROI 
            # [x1, x2, y1, y2] are the points of the rectagle around the face
            # stores as list elements for each face detected
            #  face 1{[x1, x2, y1, y2],
            #  face 1 [x1, x2, y1, y2]} when there are two faces in the image

        cv2.rectangle(gray,(x1,y1),(x2,y2),(255,0,0),2)
            # draw a rectangle around the face

        points=landmark_detector(gray,rect)
            # passing thr gray image with ROI, returns the 68 points
            # points are not numpy array
            # to covert this into numpy array use imutils - image utilities library

        points=face_utils.shape_to_np(points)
            # converting the points object into a numpy array

        myPoints=points[2:9,0]
            # ,0 - retreave x coordinate of 2-8 points

        D1=myPoints[6]-myPoints[0]
        D2=myPoints[6]-myPoints[1]
        D3=myPoints[6]-myPoints[2]
        D4=myPoints[6]-myPoints[3]
        D5=myPoints[6]-myPoints[4]
        D6=myPoints[6]-myPoints[5]

        # taking ratios to get scale invariant feature (small when far, large when close)
        d1=D2/D1
        d2=D3/D1
        d3=D4/D1
        d4=D5/D1
        d5=D6/D1

        result=algorithm.predict([[d1,d2,d3,d4,d5]])
            # [[]] why two square brackets - bcuz algorithm.predict accepts features as 2D array
        #print(results)

        img[0:50]=(255,0,0)
        # colour first 50 rows

        # include text in the frame
        cv2.putText(img,label_dict[result[0]],(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.5,[255,255,255],2)
        # img,label_dict[result[0]] - to get the first array value,
        # (10,40) - origin of the text ,
        # cv2.FONT_HERSHEY_SIMPLEX- font,
        # 1.5 - font scale ,
        # [255,255,255]- font colour,
        # 2 - letter thickness          

        

while(True):
    ret,img=camera.read()

    predict_face_type(img)
    
    cv2.imshow('Live',img)
    cv2.waitKey(1)
