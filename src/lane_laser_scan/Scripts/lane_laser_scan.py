#!/usr/bin/env python3
import rospy
from std_msgs.msg import String,Float32
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
import numpy as np
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from line_fit import line_fit, viz2, calc_curve, final_viz
from time import time
from math import tan,atan

with open(os.path.dirname(__file__)+'/calibrate_camera.p', 'rb') as f:
    save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']
pi=3.14
fov=20 #blind angle in one quadrant = 90-70; 140 total field of view
xc=(640-1)/2
yc=-480.0 - xc*tan(fov*pi/180)

def grid(img):
    img = cv2.undistort(img, mtx, dist, None, mtx)
    img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)
    img, binary_unwarped, m, m_inv = perspective_transform(img)
    xm_per_pix = 3.2/640
    ym_per_pix = 2.0/480
    height=img.shape[0]*ym_per_pix/0.1
    width=img.shape[1]*xm_per_pix/0.1
    rsz=cv2.resize(img, (int(width),int(height)), interpolation = cv2.INTER_AREA)
    rsz[rsz>0]=1
    return rsz

def laserScan(img,xc,yc,inc,amin,amax):
    img = cv2.undistort(img, mtx, dist, None, mtx)
    img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)
    img, binary_unwarped, m, m_inv = perspective_transform(img)
    xm_per_pix = 3.2/640
    ym_per_pix = 2.0/480
    # height=img.shape[0]*ym_per_pix/0.1
    # width=img.shape[1]*xm_per_pix/0.1
    # rsz=cv2.resize(img, (int(width),int(height)), interpolation = cv2.INTER_AREA)
    # rsz[rsz>0]=1
    ranges=np.ones([int(2*amax//inc)], dtype=float)*4.0
    ranges=ranges.tolist()
    for i in range(img.shape[1]): # width - columns - X
        for j in range(img.shape[0]): #height - rows - Y
            if img[j][i]:
                slope=atan((yc+j)/(xc-i))
                angle = (-1)**(slope>0)*pi/2 + atan((yc+j)/(xc-i))  #*180/pi
                r = (((yc+j)*ym_per_pix)**2 + ((xc-i)*xm_per_pix)**2)**0.5
                rangeIndex= int((angle + amax)/inc) 
                # print(i,j,rangeIndex)
                if r<ranges[rangeIndex]: ranges[rangeIndex] = r
    return ranges
                
##img = mpimg.imread('test_images/' + '2019-12-19-153528.jpg')
##t1=time()
##img=grid(img)
##print(1/(time()-t1),end=" ");print("fps")
##cv2.imshow('OG',img)
##cv2.waitKey()
##cv2.destroyAllWindows()

def talker():
    ls= LaserScan()
    ls.header.frame_id = 'map'
    inc=0.02
    amin=-pi/2 + fov*pi/180 #20 implies 70 deg field of view in one quadrant
    amax=-float(amin)
    ls.angle_increment=inc
    ls.angle_max=amax
    ls.angle_min=amin
    ls.time_increment=0.0001
    ls.scan_time=0.00001
    ls.range_min=0.5
    ls.range_max=4.0
    pub=rospy.Publisher('laser_scan_test', LaserScan,queue_size=10)
    rospy.init_node('lane_laser_scan', anonymous=True)
    rate = rospy.Rate(10)
    #cap = cv2.VideoCapture(os.path.dirname(__file__)+"/test_480.mp4")
    cap = cv2.VideoCapture(1)
    #cap.set(cv2.cv.CV_CAP_PROP_FPS, 60)
    ret,frame=cap.read()
    while ret:
        ret,frame=cap.read()
        ls.ranges=laserScan(frame,xc,yc,inc,amin,amax)
        ls.header.stamp.secs=rospy.get_rostime().secs
        ls.header.stamp.nsecs=rospy.get_rostime().nsecs
        #ls.header.stamp.sec=rospy.Time
        rospy.loginfo(str(ls))
        pub.publish(ls)
        rate.sleep()
    
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass


##def talker():
##    pub = rospy.Publisher('chatter', String, queue_size=10)
##    rospy.init_node('talker', anonymous=True)
##    rate = rospy.Rate(10) # 10hz
##    while not rospy.is_shutdown():
##        hello_str = "hello world %s" % rospy.get_time()
##        rospy.loginfo(hello_str)
##        pub.publish(hello_str)
##        rate.sleep()
##
##if __name__ == '__main__':
##    try:
##        talker()
##    except rospy.ROSInterruptException:
##        pass
##    
##

#rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 map global 10
    
