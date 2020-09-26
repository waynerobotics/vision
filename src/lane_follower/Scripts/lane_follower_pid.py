#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from os import system

previousError = -0.01 # Initiatlizing to a small amount
integral = 0

set_point = 0
kp, ki, kd = 0.5, 0, 0.01 

def pid_update(data):
    global integral, previousError, previousTime
    currentTime = rospy.get_time()
    dt = currentTime - previousTime
    ctError = data.data - set_point
    print("Cross Track Error: ",ctError)
    
    integral += ctError*dt
    derivative = (ctError - previousError)/dt
    bot.angular.z = -(kp*ctError + ki*integral + kd*derivative)
    bot.linear.x = 0.2
    pub_car.publish(bot)

    previousTime = currentTime
    previousError = ctError
    print("Publishing velocity: ", bot.linear.x,"m/s and\n \
          ", "yaw :",bot.angular.z, "\n Time step: ",dt," seconds")
    
    
def main():
    global bot, pub_car, previousTime
    rospy.sleep(5)
    bot = Twist()
    bot.linear.x = 0.1
    bot.angular.z = 0
    rospy.init_node('lane_follower_pid', anonymous=True)
    pub_car = rospy.Publisher('cmd_vel', Twist,queue_size=10)
    pub_car.publish(bot)
    previousTime = rospy.get_time() # For the first instance
    rospy.Subscriber("/laneOffset", Float32, pid_update)

    rospy.spin()

if __name__ == '__main__':
    main()