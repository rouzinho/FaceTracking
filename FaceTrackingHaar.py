## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import dlib
from both import Both
from numpy import linalg as la
import rospy
import sys
import csv
import math

from std_msgs.msg import Float64
from std_msgs.msg import Bool

pitch = Float64(0.0)
yaw = Float64(0.0)

def movePitch(p,y):
    #tmp_p = Float64(-p.data)
    #tmp_y = Float64(-y.data)
    head_pitch.publish(p)
    head_yaw.publish(y)

def center_face(x,y,w,h):
    x = x+w/2
    y = y+h/2
    return [x,y]

def calcVector(x_center,y_center,x_face,y_face,p1,y1):
    x = x_face - x_center
    y = y_face - y_center
    #x = x_center - x_face
    #y = y_center - y_face
    tmp = [x,y]
    #print("magnitude vector :")
    #print(la.norm(tmp))
    if la.norm(tmp) > 20:
        vect = tmp/la.norm(tmp)
        vect = vect/20
        p = vect[0]
        y = vect[1]
        #print("translation vector :")
        #print(vect)
        #p1 = Float64(p+p1.data)
        #y1 = Float64(y+y1.data)
        if p < 0:
            p1 = Float64(p1.data+0.01)
        else :
            p1 = Float64(p1.data-0.01)
        if y > 0:
            y1 = Float64(y1.data+0.01)
        else :
            y1 = Float64(y1.data-0.01)


    return [p1,y1]

def setPresence(presence):
    person.publish(presence)


#Create the tracker we will use
tracker = dlib.correlation_tracker()

head_yaw = rospy.Publisher("/head_pitch_controller/command", Float64, queue_size=10)
head_pitch = rospy.Publisher("/head_yaw_controller/command", Float64, queue_size=10)
person = rospy.Publisher("/hand_shake/person", Bool, queue_size=10)
present_person = Bool()
present_person.data = False

rospy.init_node('gummi', anonymous=True)
r = rospy.Rate(30)
#The variable we use to keep track of the fact whether we are
#currently using the dlib tracker
trackingFace = 0


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600

b = Both()

rectangleColor = (0,165,255)

# Start streaming
pipeline.start(config)

try:
    while True and not rospy.is_shutdown():

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        #Resize the image to 320x240
        baseImage = cv2.resize( color_image, (320, 240))
        #baseImage = image


        #Check if a key was pressed and if it was Q, then destroy all
        #opencv windows and exit the application
        pressedKey = cv2.waitKey(2)
        if pressedKey == ord('Q'):
            cv2.destroyAllWindows()
            exit(0)

        resultImage = baseImage.copy()

        gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        #Now use the haar cascade detector to find all faces in the
        #image
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)


        #For now, we are only interested in the 'largest' face, and we
        #determine this based on the largest area of the found
        #rectangle. First initialize the required variables to 0
        maxArea = 0
        x = 0
        y = 0
        w = 0
        h = 0


        #Loop over all faces and check if the area for this face is
        #the largest so far
        for (_x,_y,_w,_h) in faces:
            if  _w*_h > maxArea:
                x = _x
                y = _y
                w = _w
                h = _h
                maxArea = w*h

            #If one or more faces are found, draw a rectangle around the
            #largest face present in the picture
            if maxArea > 0 :
                cv2.rectangle(resultImage,  (x-10, y-20),
        	    		    (x + w+10 , y + h+20),
        		    	    rectangleColor,2)
            c = center_face(x,y,w,h)
            tmp = calcVector(160,120,c[0],c[1],pitch,yaw)
            pitch = tmp[0]
            yaw = tmp[1]
            movePitch(pitch,yaw)
            print("size of rectangle :")
            print(h)
            if h > 60:
                present_person.data = True
            else :
                present_person.data = False

            setPresence(present_person)

        #If we are not tracking a face, then try to detect one
        if not trackingFace:

            gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
            #Now use the haar cascade detector to find all faces
            #in the image
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)


            maxArea = 0
            x = 0
            y = 0
            w = 0
            h = 0


            for (_x,_y,_w,_h) in faces:
                if  _w*_h > maxArea:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)
                    maxArea = w*h

            #If one or more faces are found, initialize the tracker
            #on the largest face in the picture
            if maxArea > 0 :

                #Initialize the tracker
                tracker.start_track(baseImage,
                                    dlib.rectangle( x-10,
                                                    y-20,
                                                    x+w+10,
                                                    y+h+20))

                trackingFace = 1
                c = center_face(x,y,w,h)
                tmp = calcVector(160,120,c[0],c[1],pitch,yaw)
                pitch = tmp[0]
                yaw = tmp[1]
                movePitch(pitch,yaw)
                print(tmp)

        largeResult = cv2.resize(resultImage,
        		     (OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))




        # Show images
        # Stack both images horizontally
        #images = np.hstack((color_image, largeResult))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.imshow("result-image", largeResult)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
