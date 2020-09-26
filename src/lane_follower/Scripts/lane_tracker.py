#!/usr/bin/env python

import sys
import rospy
from cv2 import cv2 # To make PyLint recognize cv2
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from combined_thresh import combined_thresh
from combined_thresh_canny import combined_thresh_canny
from perspective_transform import perspective_transform
from line_fit import line_fit, viz2, calc_curve, final_viz
from time import time
import os

image_topic = '/usb_cam1/image_raw'
count = 0

class image_offset:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(image_topic,Image,self.callback)
    self.pub = rospy.Publisher("laneOffset", Float32, queue_size=10)

  def callback(self,data):
    global count
    try:
      img = self.bridge.imgmsg_to_cv2(data, desired_encoding = "bgr8")
      
      if img.dtype == 'float32':
        img = np.array(img)*255
        img = np.uint8(img)
      
      img = cv2.blur(img, (5,5))  
      img, _, _, _, _ = combined_thresh(img)
      #_, _, img = combined_thresh_canny(img)
      #img, _, _, _ = perspective_transform(img)
      cv2.imshow("Image window", img.astype(np.float32))
      count += 1
      k = cv2.waitKey(1)
      if k == 113: #q
          cv2.imwrite("saves/"+str(count)+"_igvcw.png", img)
      if k == 27: #esc
          cv2.destroyAllWindows()
      try:
        ret = line_fit(img)
        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        bottom_y = img.shape[0] - 1
        bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
        bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
        vehicle_offset = img.shape[1]/2 - (bottom_x_left + bottom_x_right)/2
        # if 350 < bottom_x_right - bottom_x_left < 600 :
        #   self.pub.publish(0.0)
        #   return 0
        
        xm_per_pix = 3.7/680 # meters per pixel in x dimension
        vehicle_offset = vehicle_offset*xm_per_pix
        self.pub.publish(vehicle_offset) # cross track error
      except TypeError:
        print("No lanes found.")
        self.pub.publish(0.0)
      #label_str = 'Vehicle offset from lane center: %.3f m' % self.vehicle_offset
      #img = cv2.putText(img, label_str, (30,70), 0, 1, (255,0,0), 2, cv2.LINE_AA)
    except CvBridgeError as e:
      print(e)
    

def main():
  rospy.init_node('lane_tracker', anonymous=True)
  ic = image_offset()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
