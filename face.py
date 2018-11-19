from numpy import linalg as LA
import cv2
#from matplotlib import pyplot as plt

def center_face(x,y,w,h):
    x = x+w/2
    y = y+h/2
    return [x,y]

def calcVector(x_center,y_center,x_face,y_face):
    x = x_face - x_center
    y = y_face - y_center
    tmp = [x,y]
    vect = tmp/LA.norm(tmp)
    return vect

t = center_face(45,45,180,200)
print(t)
e = calcVector(320,240,t[0],t[1])
print(e/10)
