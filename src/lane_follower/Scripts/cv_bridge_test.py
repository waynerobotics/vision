#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    rospy.Subscriber("/front_camera/image_color",Image,self.callback)
    self.pub = rospy.Publisher('cross_track_errors', Float32, queue_size=10)

  def callback(self,data):
    try:

      img = self.bridge.imgmsg_to_cv2(data, desired_encoding = "bgr8")
      #error = cross_track_error(img)
      #self.pub.publish(error)

    except CvBridgeError as e:
      print(e)
    cv2.imshow("Image window", img)
    k = cv2.waitKey(1)
    if k == 27: # escape key
        cv2.destroyAllWindows()

if __name__ == "__main__":
  rospy.init_node('cv_bridge_test', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()