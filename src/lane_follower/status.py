#!/usr/bin/env python3

from lane_follower.msg import VeronicaStatusReport
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import rospy

imu, camera, gps, lidar, driver = 0,1,2,3,4

class status:

    def __init__(self):
        self.reporter = VeronicaStatusReport()
        self.status_string = String()

        self.reporter.IMU_STATUS = False
        self.reporter.CAMERA_STATUS = False
        self.reporter.GPS_STATUS = False
        self.reporter.LIDAR_STATUS = False
        self.reporter.MOTOR_DRIVER_STATUS = False

        self.pub = rospy.Publisher("robot_status_report", VeronicaStatusReport, queue_size=10)
        self.led_pub = rospy.Publisher("status_led_strip", String, queue_size=10)

        self.status_string.data = "00000"
        self.pub.publish(self.reporter)
        self.led_pub.publish(self.status_string)

        self.arduino_handshake = rospy.Subscriber("arduino_nano_handshake", Bool, self.arduino_nano)
        self.camera = rospy.Subscriber("/usb_cam1/image_raw", Image, self.camera_status)
        #self.lidar = rospy.Subscriber("/scan", LaserScan, self.lidar_status)

    def camera_status(self, msg):
        self.reporter.CAMERA_STATUS = True
        self.status_string.data = self.status_string.data[:camera] + "1" + self.status_string.data[camera+1:]

        self.pub.publish(self.reporter)
        self.led_pub.publish(self.status_string)

        #IMU, LIDAR, GPS Simulator -- REMOVE block when physical testing
        rospy.sleep(2.5)
        self.status_string.data = self.status_string.data[:imu] + "1" + self.status_string.data[imu+1:]
        self.led_pub.publish(self.status_string)
        rospy.sleep(2)
        self.status_string.data = self.status_string.data[:lidar] + "1" + self.status_string.data[lidar+1:]
        self.led_pub.publish(self.status_string)
        rospy.sleep(1)
        self.status_string.data = self.status_string.data[:gps] + "1" + self.status_string.data[gps+1:]
        self.led_pub.publish(self.status_string)
        rospy.sleep(1)
        self.status_string.data = self.status_string.data[:driver] + "1" + self.status_string.data[driver+1:]
        self.led_pub.publish(self.status_string)
        

    def arduino_nano(self, msg):
        if(msg.data) == True:
            self.reporter.MOTOR_DRIVER_STATUS = True
            self.status_string.data = self.status_string.data[:driver] + "1" + self.status_string.data[driver+1:]
        
        self.pub.publish(self.reporter)
        self.led_pub.publish(self.status_string)

    def lidar_status(self, msg):
        self.reporter.LIDAR_STATUS = True
        self.status_string.data = self.status_string.data[:lidar] + "1" + self.status_string.data[lidar+1:]

        self.pub.publish(self.reporter)   
        self.led_pub.publish(self.status_string)


if __name__ == "__main__":
    rospy.init_node("status", anonymous=True)
    s = status()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
