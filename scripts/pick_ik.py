#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
from math import *
from time import sleep
import sys
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import roslib
from std_msgs.msg import String, Int32MultiArray
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from moveit_msgs.msg import Grasp, GripperTranslation, PlaceLocation
from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState
import actionlib
import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import WrenchStamped, Vector3
import tf
from tf.transformations import *
from math import pi
from icl_phri_robotiq_control.robotiq_utils import *
from inverseKinematicsUR5 import InverseKinematicsUR5, transformRobotParameter
from copy import deepcopy
from std_msgs.msg import Int32
import cv2


JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
SPEED = 3
pro_pub = rospy.Publisher('/netsend', Int32MultiArray, queue_size=1)
# , [0.340,-0.261], [449, 103]
def netsend(msg, flag=-1, need_unpack=True):
    global pro_pub, gesture_id
    if msg:
        if flag != 1:
            rospy.loginfo("flag is {}. msg is {}".format(flag, msg))
        if need_unpack:
            send = []
            for i in range(len(msg)):
                send.append(int(msg[i][0]))
                send.append(int(msg[i][1]))
            a = deepcopy(send)
            a.append(flag)
        else:
            a = deepcopy(msg)
            a.append(flag)
        pro_pub.publish(Int32MultiArray(data=a))
    
def coord_converter(x, y):
    # pts1 = np.array([[138, 133], [281, 133], [429, 130], [134, 271], [280, 251], [417, 260], [137, 391], [274, 381]])
    # pts2 = np.array([[0.391, -0.319], [0.557, -0.337], [0.719, -0.355], [0.386, -0.496], [0.557, -0.489], [0.707, -0.470], [0.379, -0.640], [0.541, -0.641]])
    #pts1 = np.array([[(118, 84), (131, 239), (134, 369), (295, 354), (292, 85), (304, 227), (444, 237), (465, 76)]])
    #pts2 = np.array([[(0.369, -0.309), (0.399, -0.497), (0.376, -0.646), (0.589, -0.643), (0.568, -0.310), (0.589, -0.495),(0.759, -0.495), (0.792, -0.301)]])
    # pts1 = np.array([[(68, 57), (76, 217), (78, 345), (271, 220), (274, 358), (274, 147), (433, 338), (440, 232)]])
    # pts2 = np.array([[(-0.387,-0.238), (-0.341, -0.461), (-0.306, -0.652), (0.011, -0.448), (0.047, -0.637),(-0.013, -0.340),(0.361, -0.603),(0.357, -0.450)]])
    # pts2 = np.array([[(-0.406, -0.398), (-0.389, -0.544), (-0.371, -0.735), (-0.004, -0.381), (0.017, -0.556), (0.026, -0.720), (0.354, -0.373), (0.372, -0.507), (0.352, -0.674)]])
    # pts1 = np.array([[49, 124], [61, 231], [71, 360], [274, 117], [283, 235], [287, 352], [474, 105], [466, 202], [473, 325]])
    pts2 = np.array([[(-0.343, -0.242), (-0.342, -0.518), (-0.339, -0.751), (0.040, -0.233), (0.015, -0.501), (0.061, -0.768), (0.437, -0.235), (0.451, -0.522), (0.448, -0.755)]])
    pts1 = np.array([[96, 28], [95, 226], [98, 388], [318, 24], [313, 213], [325, 393], [536, 27],[537, 228],[533, 390]])
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)
    solusion = np.matmul(M,np.array([x, y, 1]))
    solusion = solusion/solusion[2]
    return solusion[0], solusion[1]
#tool0 is ee
class pick_place:
    def __init__(self):
        #/vel_based_pos_traj_controller/
        self.client = actionlib.SimpleActionClient('icl_phri_ur5/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory = JointTrajectory()
        self.goal.trajectory.joint_names = JOINT_NAMES
        print ("Waiting for server...")
        self.client.wait_for_server()
        print ("Connected to server")
        joint_states = rospy.wait_for_message("joint_states", JointState)
        print(joint_states)
        joint_states = list(deepcopy(joint_states).position)
        del joint_states[-1]
        self.joints_pos_start = np.array(joint_states)
        print ("Init done")
        self.listener = tf.TransformListener()
        self.Transformer = tf.TransformerROS()

        joint_weights = [12,5,4,3,2,1]
        self.ik = InverseKinematicsUR5()
        self.ik.setJointWeights(joint_weights)
        self.ik.setJointLimits(-pi, pi)
        self.sub = rospy.Subscriber('/target_position', Int32MultiArray, self.pickplace_cb)
        self.sub_cancel = rospy.Subscriber('/voice_command', Int32, self.cancel_cb)

        self.gripper_ac = RobotiqActionClient('icl_phri_gripper/gripper_controller')
        self.gripper_ac.wait_for_server()
        self.gripper_ac.initiate()
        self.gripper_ac.send_goal(0.10)
        self.gripper_ac.wait_for_result()
        self.cancel = False

    def define_grasp(self, position):
        quat = tf.transformations.quaternion_from_euler(3.14, 0, -3.14)
        dest_m = self.Transformer.fromTranslationRotation(position, quat) 
        return dest_m

    def move(self, dest_m):
        #current_m = transformRobotParameter(self.joints_pos_start)
        qsol = self.ik.findClosestIK(dest_m,self.joints_pos_start)
        
        if qsol is not None and not self.cancel:
            if qsol[0] < 0:
                qsol[0] += pi
            else:
                qsol[0] -= pi
            self.goal.trajectory.points = [
                JointTrajectoryPoint(positions=self.joints_pos_start.tolist(), velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
                JointTrajectoryPoint(positions=qsol.tolist(), velocities=[0]*6, time_from_start=rospy.Duration(SPEED)),
            ]
            #print('start: ' + str(self.joints_pos_start.tolist()))
            #print('goal: ' + str(qsol.tolist()))
            try:
                self.client.send_goal(self.goal)
                self.joints_pos_start = qsol
                self.client.wait_for_result()
            except:
                raise
        elif qsol is None:
            rospy.loginfo("fail to find IK solution")
        elif self.cancel:
            rospy.logwarn("this goal canceled")
    
    def single_exuete(self, position, mode):
        offset = 0.015
        offset1 = 0.005
        position_copy = deepcopy(position)
        if position_copy[0] < 0:
            position_copy += [0.19]
            position_copy[1] = position_copy[1]
        else:
            position_copy += [0.192]
            position_copy[1] = position_copy[1]
        # position_copy[1] = position_copy[1] + offset
        position_copy[0] = position_copy[0] + offset1
        pre_position = self.define_grasp([position_copy[0], position_copy[1], position_copy[2] + 0.1])
        post_position = self.define_grasp([position_copy[0], position_copy[1], position_copy[2] + 0.1])
        grasp_position = self.define_grasp(position_copy)
        rospy.loginfo("let's go to the pre location")
        self.move(pre_position)
        #rospy.sleep(1)
        rospy.loginfo("let's do this")
        self.move(grasp_position)
        #rospy.sleep(1)
        if mode == "pick":
            self.gripper_ac.send_goal(0)
        elif mode == "place":
            self.gripper_ac.send_goal(0.10)
        self.gripper_ac.wait_for_result()
        rospy.loginfo("move out")
        self.move(post_position)
        #rospy.sleep(1)

    def pair_exuete(self, pick_position, place_position):
        rospy.loginfo("here we go pair")
        if pick_position and place_position:
            self.single_exuete(pick_position, "pick")
            self.single_exuete(place_position, "place")
            #rospy.sleep(1)
            
            #rospy.sleep(1)
    
    def pickplace_cb(self, msg):
        #print(msg)
        print(msg.data)
        self.cancel = False
        a = list(msg.data)
        mean_x = np.mean([a[i] for i in range(0, len(a)-2, 2)])
        mean_y = np.mean([a[i] for i in range(1, len(a)-2, 2)])
        num_goals = (len(msg.data) -2)/2
        rospy.loginfo("there is {} goals".format(num_goals))
        for i in range(0, len(a)-2, 2):
            pick_x, pick_y = coord_converter(msg.data[i], msg.data[i+1])
            leeway_x = int(msg.data[i] - mean_x)
            leeway_y = int(msg.data[i+1] - mean_y)
            place_x, place_y = coord_converter(msg.data[-2] + leeway_x, msg.data[-1] + leeway_y)
            print(pick_x, pick_y)
            print(place_x, place_y)
            self.pair_exuete([pick_x, pick_y], [place_x, place_y])
            if self.cancel:
                self.cancel = False

                joint_states = rospy.wait_for_message("joint_states", JointState)
                joint_states = list(deepcopy(joint_states).position)
                del joint_states[-1]
                self.joints_pos_start = np.array(joint_states)

                self.client.wait_for_result()
                netsend([777, 888], need_unpack=False,flag=-99)
                # rospy.loginfo("go to init because goal canceled")
                # rest_position = self.define_grasp([0.405, 0.010, 0.342])
                # self.move(rest_position)
                break
        rospy.loginfo("let's go and get some rest")
        rest_position = self.define_grasp([0.405, 0.010, 0.342])
        self.move(rest_position)
                
    
    def cancel_cb(self, msg):
        if msg.data == -9:
            rospy.logwarn("canceling goals")
            self.client.cancel_all_goals()
            self.cancel = True







if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)
    task = pick_place()
    rospy.spin()
    
    # pick_x, pick_y = coord_converter(419, 219)
    # place_x, place_y = coord_converter(275, 352)
    # task.pair_exuete([pick_x, pick_y], [place_x, place_y])
    #print(coord_converter(449, 103))
