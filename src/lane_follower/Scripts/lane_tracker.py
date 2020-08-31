#!/usr/bin/env python

import sys
import rospy
from cv2 import cv2 # To make PyLint recognize cv2
from std_msgs.msg import Float32, String, Int8
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from line_fit import line_fit, viz2, calc_curve, final_viz
from time import time
from nav_msgs.msg import Odometry
import os

image_topic = '/front_camera/image_raw'
sign_detection_topic = "/darknet_ros/bounding_boxes"
count = 0  # For saved image filenames in saves/
LANE_KEEP, STOP, LEFT_TURN, RIGHT_TURN, KEEP_STRAIGHT = 0,1,2,3,4

class image_offset:

  def __init__(self):
    self.bridge = CvBridge()
    self.vehicleState = LANE_KEEP # Default State
    rospy.Subscriber(image_topic,Image,self.callback)
    rospy.Subscriber("/vehicle/perfect_gps/utm", Odometry, self.distance_to_line)
    rospy.Subscriber(sign_detection_topic, BoundingBoxes, self.object_detection)
    rospy.Subscriber("/vehicle/state", Int8, self.set_vehicle_state)
    self.pub = rospy.Publisher("/laneOffset", Float32, queue_size=10)
    self.ad = rospy.Publisher("/stop_line_approach_distance", Float32, queue_size=10)
    plt.ion() # Interactive Plot
    
  def distance_to_line(self, data):
    self.currentDistance = 500006.41 - data.pose.pose.position.x
    #print(self.currentDistance)
  
  def randomshit(self):
    self.x = 123

  def set_vehicle_state(self, data):
    self.vehicleState = data.data

  def object_detection(self, data):
    if(data.bounding_boxes[0].probability > 0.90):
      self.object = data.bounding_boxes[0].Class  # Returns detected object as string
  
  def stop_line_approach_distance(self,img): # Publish pixel distance to stop line.
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 50, 300, 50)
    distance = []
    try:
      for i in lines:
        for x1, y1, x2, y2 in i:
          angle = np.arctan2(y1-y2, x1-x2)*(180/3.14)
          if abs(angle) < 180 and abs(angle) >150:
            distance.append((y1+y2)/2)    
    except TypeError: # This will happen if there are no lines at all in the image.
      self.ad.publish(10000)  # Publishing some random safe distance = 30 meters
    
    if len(distance) == 0: # This will happen if no stop line is detected
      self.ad.publish(10000)
    else:
      self.approach_distance_pixels = img.shape[0]-np.mean(distance)
      self.ad.publish(self.approach_distance_pixels)
    # print(self.currentDistance/self.approach_distance_pixels) # Proportional Constant
    # plt.scatter(self.approach_distance_pixels, self.currentDistance) # Scatter Plot
    # plt.pause(0.05)
    # plt.draw()

  def callback(self,data):
    global count
    try:
      img = self.bridge.imgmsg_to_cv2(data, desired_encoding = "bgr8")
      if img.dtype == 'float32':
        img = np.array(img)*255
        img = np.uint8(img)
      
      img = cv2.blur(img, (5,5))  
      img, _, _, _, _ = combined_thresh(img)
      img, _, _, _ = perspective_transform(img)

      self.stop_line_approach_distance(img)
      # if (self.object == 'stop sign'):
      #   self.stop_line_approach_distance(img)
      
      if (self.vehicleState == LANE_KEEP):
        ret = line_fit(img)
        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        bottom_y = img.shape[0] - 1
        bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
        bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
        vehicle_offset = img.shape[1]/2 - (bottom_x_left + bottom_x_right)/2
        xm_per_pix = 3.7/680 # meters per pixel in x dimension
        vehicle_offset = vehicle_offset*xm_per_pix
        self.pub.publish(vehicle_offset) # cross track error
        #label_str = 'Vehicle offset from lane center: %.3f m' % self.vehicle_offset
        #img = cv2.putText(img, label_str, (30,70), 0, 1, (255,0,0), 2, cv2.LINE_AA)
    except CvBridgeError as e:
      print(e)
    
    # cv2.imshow("Image window", (img))
    # count += 1
    # k = cv2.waitKey(1)
    # if k == 113: #q
    #     cv2.imwrite("saves/"+str(count)+"_igvcw.png", img)
    # if k == 27: #esc
    #     cv2.destroyAllWindows()

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
