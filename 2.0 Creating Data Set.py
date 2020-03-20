import os
import cv2
import dlib
from imutils import face_utils

datapath='Face Shapes'
    # data path is given, folder name

labels=os.listdir(datapath)
    # os.listdir(datapath) - returns directories in the given path

print(labels)

##print(os.getcwd())
##new=os.mkdir('New Folder')
##print(os.listdir(new))


label_dict={}

data=[]
target=[]
# declared two empty lists


face_detector=dlib.get_frontal_face_detector()
    # a pre-trained faced detector algorothm in dlib

landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # a pre-trained external shape_predictor_68_face_landmarks

for i in range(len(labels)):
    label_dict[labels[i]]=i

print(label_dict)


def cal_distance(img,label): # function to calculate distances
    
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


        data.append([d1,d2,d3,d4,d5]) # 1d array with 5 columns
        target.append(label)


for label in labels: # runs 6 times for 6 shapes as there are 6 type folders
    imgs_path=os.path.join(datapath,label) # create new path name
    print(imgs_path)

    imgs_name=os.listdir(imgs_path) # images in each new path 
    print(imgs_name)

    for img_name in imgs_name:
        im_path=os.path.join(imgs_path,img_name)
        img=cv2.imread(im_path)

        cv2.imshow('Live',img)
        cv2.waitKey(100)


        cal_distance(img,label_dict[label])
        # labels need a number to represent it other than strings

        

print(data)
print(target)


# going to save these data in the harddisk from the ram where they are stored when the program is runing


import numpy as np

# saving numpy array as a physical file in the harddisk, this is portable

np.save('data',data)
    # file name -'data', data
    # images are no longer required as the data set is saved in a physical file
np.save('target',target)



