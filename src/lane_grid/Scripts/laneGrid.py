#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
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

with open('/home/ringo/catkin_ws/src/lane_grid/Scripts/calibrate_camera.p', 'rb') as f:
    save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

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

##img = mpimg.imread('test_images/' + '2019-12-19-153528.jpg')
##t1=time()
##img=grid(img)
##print(1/(time()-t1),end=" ");print("fps")
##cv2.imshow('OG',img)
##cv2.waitKey()
##cv2.destroyAllWindows()

def talker():
    t=0
    og = OccupancyGrid()
    og.header.frame_id = 'map';
    og.info.resolution = .1 #meters per cell
    og.info.width = 32
    og.info.height = 20
    pub=rospy.Publisher('lane_grid_test', OccupancyGrid,queue_size=10)
    rospy.init_node('laneGrid', anonymous=True)
    rate = rospy.Rate(10)
    cap = cv2.VideoCapture("/home/ringo/catkin_ws/src/lane_grid/Scripts/test_480.mp4")
    #cap.set(cv2.cv.CV_CAP_PROP_FPS, 60)
    ret,frame=cap.read()
    while ret:
        ret,frame=cap.read()
        #cv2.imshow('OG',grid(frame))
        og.data = np.array(grid(frame).flatten(),dtype=np.uint8)
        og.info.origin.position.x = 0.01*t
        rospy.loginfo(str(og))
        pub.publish(og)
        t=t+1 #**remove later
        rate.sleep()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

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
    
