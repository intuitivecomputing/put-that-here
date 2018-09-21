#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
import time

rospy.init_node('fake_publisher')
pub = rospy.Publisher('/voice_command', Int32, queue_size=1)
start = time.time()
while True:
    num = raw_input('please enter \n')
    pub.publish(Int32(data=int(num)))
    # if time.time() - start == 5.0:
    #     num = 1
    #     #num = raw_input('please enter \n')
    #     print "1"
    #     pub.publish(Int32(data=int(num)))
    # elif time.time() - start == 10.0:
    #     num = 2
    #     #num = raw_input('please enter \n')
    #     pub.publish(Int32(data=int(num)))
    #     print "2"
    # elif time.time() - start == 15.0:
    #     break
    if not rospy.is_shutdown:
        break
#rospy.spin()