#!/usr/bin/env python3

from lane_follower.msg import VeronicaStatusReport
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
import rospy

class status:

    def __init__(self):
        self.reporter = VeronicaStatusReport()
        self.pub = rospy.Publisher("robot_status_report", VeronicaStatusReport, queue_size=10)
        self.arduino_handshake = rospy.Subscriber("arduino_nano_handshake", Bool, self.arduino_nano)
        self.camera = rospy.Subscriber("/usb_cam1/image_raw", Image, self.camera_status)
        self.lidar = rospy.Subscriber("/scan", LaserScan, self.lidar_status)
        self.pub.publish(self.reporter)

    def camera_status(self, msg):
        self.reporter.CAMERA_STATUS = True
        self.pub.publish(self.reporter)

    def arduino_nano(self, msg):
        if(msg.data) == True:
            self.reporter.MOTOR_DRIVER_STATUS = True
        self.pub.publish(self.reporter)

    def lidar_status(self, msg):
        self.reporter.LIDAR_STATUS = True
        self.pub.publish(self.reporter)   


if __name__ == "__main__":
    rospy.init_node("status", anonymous=True)
    s = status()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
