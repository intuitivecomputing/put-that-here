#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
from constant import *
from joblib import load
from utils import side_finder, test_insdie, cache
from hand_tracking import hand_tracking
from shapely.geometry import Polygon
from math import sqrt
from copy import deepcopy
import rospy
from std_msgs.msg import Int32MultiArray, Int32, String
import socket
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from PIL import Image, ImageTk
from torchvision import transforms
from torch.autograd import Variable
from utils import Net
from collections import deque, Counter
from edm import classifier

distant = lambda (x1, y1), (x2, y2) : sqrt((x1 - x2)**2 + (y1 - y2)**2)
voice_flag = 0
color_flag = None
pro_pub = rospy.Publisher('/netsend', Int32MultiArray, queue_size=1)

def warp_img(img):
    #pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
    pts1 = np.float32([[212,140],[585,127],[206,354],[595,362]])
    pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(640,480))
    return dst

def camrectify(frame):
        mtx = np.array([
            [509.428319, 0, 316.944024],
            [0.000000, 508.141786, 251.243128],
            [0.000000, 0.000000, 1.000000]
        ])
        dist = np.array([
            0.052897, -0.155430, 0.005959, 0.002077, 0.000000
        ])
        return cv2.undistort(frame, mtx, dist)


def get_objectmask(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, Green_low, Green_high)
    hand_mask = cv2.inRange(hsv, Hand_low, Hand_high)
    hand_mask = cv2.dilate(hand_mask, kernel = np.ones((11,11),np.uint8))
    skin_mask = cv2.inRange(hsv, Skin_low, Skin_high)
    skin_mask = cv2.dilate(skin_mask, kernel = np.ones((11,11),np.uint8))
    thresh = 255 - green_mask
    thresh = cv2.subtract(thresh, hand_mask)
    thresh = cv2.subtract(thresh, skin_mask)
    thresh[477:, 50:610] = 0
    return thresh



# def netsend(msg, localhost="10.194.47.21", port=6868, flag=-1, need_unpack=True):
def netsend(msg, flag=-1, need_unpack=True):
    global pro_pub
    if msg:
        if flag != -1:
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


# def filter_dest(ls):
#     new_ls = []
#     for i in range(len(ls)):
#         cx, cy = ls[i]
#         for j in range(i+1, len(ls)):
#             lx, ly = ls[j]
#             if distant((cx, cy), (lx, ly)) < 10:
#                 break
        
                
        





class control_gui():
    def __init__(self):
        rospy.init_node('control_group')
        self.cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('G2_01123.avi',fourcc, 20.0, (640,480))
        cv2.namedWindow('gui')
        cv2.setMouseCallback('gui',self.gui_callback)
        self.command = []
        self.selected = []
        self.location = []
        self.gui_img = np.zeros((130,640,3), np.uint8)
        cv2.circle(self.gui_img,(150,50),30,(0,0,255),-1)
        cv2.putText(self.gui_img,"GO!!",(120,110),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,253))
        cv2.circle(self.gui_img,(450,50),30,(0,0,254),-1)
        cv2.putText(self.gui_img,"Cancel",(400,110),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,253))
        self.pub = rospy.Publisher('/target_position', Int32MultiArray, queue_size=1)

    def update(self):
        OK, origin = self.cap.read()
        if OK:
            rect = camrectify(origin)
            self.out.write(rect)
            warp = warp_img(rect)
            thresh = get_objectmask(warp)
            #cv2.imshow("dd", thresh)
            self.image = warp.copy()
            draw_img1 = warp.copy()
            self.get_bound(draw_img1, thresh)
            for i, (cx, cy) in enumerate(self.selected):
                cv2.circle(draw_img1, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(draw_img1, str(len(self.selected)),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,123,123), 3)
            if len(self.location) > 0:
                cx, cy = self.location
                cv2.circle(draw_img1, (cx, cy), 5, (0, 255, 0), -1)
            self.temp_surface = np.vstack((draw_img1, self.gui_img))
            cv2.imshow('gui', self.temp_surface)

    def gui_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP and len(self.location) == 0:
            ind = test_insdie((x, y), self.boxls)
            if ind is not None:
                cx, cy = self.surfacels[ind]
                self.command.append(cx)
                self.command.append(cy)
                self.selected.append((cx, cy))
                
                rospy.loginfo("append target : {}, {}".format(cx, cy))
            elif ind is None and len(self.selected) > 0:
                self.location.append(x)
                self.location.append(y)
                rospy.loginfo("append destination : {}, {}".format(x, y))
        
        if event == cv2.EVENT_LBUTTONUP and (self.temp_surface[y, x] == np.array([0, 0, 255])).all():
            #netsend(self.location, flag=2, need_unpack=False)
            ls = []
            for cx, cy in set(self.selected):
                ls.append(cx)
                ls.append(cy)
            ls += self.location
            rospy.loginfo("publishing msg : {}".format(ls))
            netsend(ls[:-2], flag=1, need_unpack=False)
            rospy.sleep(0.3)
            netsend(ls[-2:], flag=2, need_unpack=False)
            self.pub.publish(Int32MultiArray(data=ls))
            self.command = []
            self.selected = []
            self.location = []
        
        if event == cv2.EVENT_LBUTTONUP and (self.temp_surface[y, x] == np.array([0, 0, 254])).all():
            rospy.loginfo("Cancel all info")
            #netsend([777, 888], need_unpack=False,flag=0)
            self.command = []
            self.selected = []
            self.location = []
                
       
    def get_bound(self, image, object_mask):
        self.surfacels = []
        self.boxls = []
        (_,object_contours, object_hierarchy)=cv2.findContours(object_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(object_contours) > 0:
            for i , contour in enumerate(object_contours):
                area = cv2.contourArea(contour)
                if area>250 and area < 800 and object_hierarchy[0, i, 3] == -1:					
                    M = cv2.moments(contour)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    x,y,w,h = cv2.boundingRect(contour)
                    self.surfacels.append((int(x+w/2), int(y+h/2)))
                    self.boxls.append((x, y, w, h))
        if len(self.boxls) > 0:
            boxls_arr = np.array(self.boxls)
            self.boxls = boxls_arr[boxls_arr[:, 0].argsort()].tolist()
            sur_array = boxls_arr = np.array(self.surfacels)
            self.surfacels = sur_array[boxls_arr[:, 0].argsort()].tolist()
        for x, y, w, h in self.boxls:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
            

    

    def __del__(self):
        self.cap.release()


if __name__ == '__main__':
    control = control_gui()
    while True:
        control.update()
        k = cv2.waitKey(1) & 0xFF # large wait time to remove freezing
        if k == 113 or k == 27:
            break
    cv2.destroyAllWindows()

