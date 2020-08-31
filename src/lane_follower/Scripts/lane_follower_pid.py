#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
from os import system
from dataspeed_ulc_msgs.msg import UlcCmd

previousError = -0.01 # Initiatlizing to a small amount
integral = 0

set_point = 0
kp, ki, kd = 0.1, 0, 0.01 

def pid_update(data):
    global integral, previousError, previousTime
    currentTime = rospy.get_time()
    dt = currentTime - previousTime
    ctError = data.data - set_point
    print("Cross Track Error: ",ctError)
    
    integral += ctError*dt
    derivative = (ctError - previousError)/dt
    car.yaw_command = kp*ctError + ki*integral + kd*derivative
    car.linear_velocity = 1.0
    pub_car.publish(car)

    previousTime = currentTime
    previousError = ctError
    # print("Publishing velocity: ", car.linear_velocity,"m/s and\n \
    #       ", "yaw :",car.yaw_command, "\n Time step: ",dt," seconds")
    
    
def main():
    global car, pub_car, previousTime
    #system("rostopic pub -r 10 /vehicle/brake_cmd dbw_fca_msgs/BrakeCmd ""{pedal_cmd_type: 2edal_cmd: 0.5,enable: 1,clear: 1,ignore: 1,count: 100}"")
    rospy.sleep(5)
    #system("rosnode kill /path_following")
    car = UlcCmd()
    car.clear = False
    car.enable_pedals = True
    car.enable_steering = True
    car.linear_velocity = 0.01
    car.shift_from_park = False
    car.enable_shifting = False
    car.lateral_accel = 0.5
    car.linear_accel = 0.1
    car.linear_decel = 5.0
    car.yaw_command = 0.00
    car.angular_accel = 0.0
    car.steering_mode=0

    rospy.init_node('lane_follower_pid', anonymous=True)
    pub_car = rospy.Publisher('/vehicle/ulc_cmd', UlcCmd,queue_size=10)
    pub_car.publish(car)
    previousTime = rospy.get_time() # For the first instance
    rospy.Subscriber("/laneOffset", Float32, pid_update)
    #system("rosnode kill /path_following")

    rospy.spin()

if __name__ == '__main__':
    main()