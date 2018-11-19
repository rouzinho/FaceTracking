#!/usr/bin/env python

import rospy
import sys
import csv
import math

from std_msgs.msg import Bool
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

#from gummi_interface.gummi import Gummi

class Both:
    def __init__(self):


        self.head_yaw = rospy.Publisher("/head_yaw_controller/command", Float64, queue_size=10)
        self.head_pitch = rospy.Publisher("/head_pitch_controller/command", Float64, queue_size=10)

        #self.pwm1_pub = rospy.Publisher("~pwm1", UInt16,  queue_size=10)


        # the JointState object is created here to save time later


    def movePitch(self,mes):
        self.head_pitch.publish(mes)
        self.head_yaw.publish(mes)
        #mes.data = mes.data+0.1



def main():

    rospy.init_node('gummi', anonymous=True)
    r = rospy.Rate(60)

    #hand_shake = HandShake()
    both = Both()


    time_counter = 1

    dat1 = Float64(0.25)
    dat2 = Float64(-0.25)
    dat3 = Float64(0.0)
    dat4 = Float64(0.15)

    dat1 = Float64(dat1.data+dat3.data)
    print(dat1)

if __name__ == '__main__':
    main()
