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
gesture_id = None

def warp_img(img):
    #pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
    pts1 = np.float32([[213,140],[581,129],[206,355],[594,360]])
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

def get_crit(mask):
    (_,contours, hierarchy)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    crit = None
    for i , contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area and hierarchy[0, i, 3] == -1:
            max_area = area
            crit = area
    return crit

def get_objectmask(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, Green_low, Green_high)
    hand_mask = cv2.inRange(hsv, Hand_low, Hand_high)
    hand_mask = cv2.dilate(hand_mask, kernel = np.ones((15,15),np.uint8))
    skin_mask = cv2.inRange(hsv, Skin_low, Skin_high)
    skin_mask = cv2.dilate(skin_mask, kernel = np.ones((11,11),np.uint8))
    thresh = 255 - green_mask
    thresh = cv2.subtract(thresh, hand_mask)
    thresh = cv2.subtract(thresh, skin_mask)
    thresh[477:, 50:610] = 0
    return thresh

def get_blue_objectmask(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, Blue_low, Blue_high)
    blue_mask = cv2.dilate(blue_mask, kernel = np.ones((11,11),np.uint8))
    return blue_mask

def get_green_objectmask(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, Green_low, Green_high)
    green_mask = cv2.dilate(green_mask, kernel = np.ones((11,11),np.uint8))
    return green_mask

def get_yellow_objectmask(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, Yellow_low, Yellow_high)
    yellow_mask = cv2.dilate(yellow_mask, kernel = np.ones((11,11),np.uint8))
    return yellow_mask

# def get_handmask(frame):
#     blur = cv2.blur(frame,(3,3))
#     hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, Hand_low, Hand_high)
#     kernel_square = np.ones((15,15),np.uint8)
#     mask = cv2.dilate(mask,kernel_square,iterations = 4)
#     return mask
def get_handmask(frame, center):
    surface = np.zeros((480, 640), dtype=np.uint8)
    temp_center = (center[0], center[1] - 30)
    cv2.circle(surface, temp_center, 70, (255), -1)
    return surface

def get_k_dis((x1, y1), (x2, y2), (x, y)):
    coord = ((x, y), (x1, y1), (x2, y2))
    return Polygon(coord).area / distant((x1, y1), (x2, y2))

# def netsend(msg, localhost="10.194.55.236", port=6868, flag=-1, need_unpack=True):
#     if msg:
#         rospy.loginfo("send tcp msg: {}".format(msg))
#         if need_unpack:
#             send = []
#             for i in range(len(msg)):
#                 send.append(int(msg[i][0]))
#                 send.append(int(msg[i][1]))
#             a = deepcopy(send)
#             a.append(flag)
#         else:
#             a = deepcopy(msg)
#             a.append(flag)
#         for i in range(len(a)):
#             client = socket.socket()
#             client.connect((localhost, port))
#             line = str(int(a[i]))
#             #print(line)
#             client.send(line.encode("utf-8"))
#             client.close()

    # pass
def netsend(msg, flag=-1, need_unpack=True):
    global pro_pub, gesture_id
    if msg:
        if flag != -1 and flag != -19 and flag != -18:
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
    # pass






class temp_tracking():
    global gesture_id
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('G3_01123.avi',fourcc, 20.0, (640,480))
        self.hand_mask = []
        self.trigger = False
        self.after_trigger = False
        if torch.cuda.is_available():
            self.net = Net().cuda()
        else:
            self.net = Net()
        self.net.load_state_dict(torch.load(f='/home/intuitivecompting/catkin_ws/src/ur5/ur5_with_gripper/icl_phri_robotiq_control/src/model'))
        self.last_select = None
        self.tip_deque = deque(maxlen=20)
        self.tip_deque1 = deque(maxlen=20)
        self.tip_deque2 = deque(maxlen=20)
        self.mode = None
        self.center = None
        self.onehand_center = None
        self.two_hand_mode = None
        self.pick_center = None
        self.gesture_mode = None
        self.pick_tip = None

    def test(self, box ,draw_img):
        global gesture_id
        net = self.net
        frame = self.image.copy()
        preprocess = transforms.Compose([transforms.Resize((50, 50)),
                                                    transforms.ToTensor()])
        #preprocess = transforms.Compose([transforms.Pad(30),
         #                                             transforms.ToTensor()])
        x,y,w,h = box
        temp = frame[y:y+h, x:x+w, :]
        #temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
        temp = cv2.blur(temp,(5,5))
        hsv = cv2.cvtColor(temp,cv2.COLOR_BGR2HSV)
        temp = cv2.inRange(hsv, Hand_low, Hand_high)
        image = Image.fromarray(temp)
        img_tensor = preprocess(image)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor).cuda()
        if torch.cuda.is_available():
            img_variable = Variable(img_tensor).cuda()
            out = np.argmax(net(img_variable).cpu().data.numpy()[0])
            #print(np.max(net(img_variable).cpu().data.numpy()[0]))
        else:
            img_variable = Variable(img_tensor)
            out = np.argmax(net(img_variable).data.numpy()[0])
            # if np.max(net(img_variable).cpu().data.numpy()[0]) > 0.3:
            #     out = np.argmax(net(img_variable).cpu().data.numpy()[0])
            # else:
            #     out = -1
        #cv2.rectangle(draw_img,(x,y),(x+w,y+h),(255, 0, 0),2)
        #cv2.putText(draw_img,str(out + 1),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
        gesture_id = int(out +1)
        return int(out) + 1

    def get_current_frame(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(0)
        OK, origin = self.cap.read()
        if OK:
            rect = camrectify(origin)
            warp = warp_img(rect)
            return warp.copy()
    

    def update(self):
        '''
        gesture flag for distinguish different scenario
        '''
        global color_flag
        OK, origin = self.cap.read()

        x = None
        if OK:
            #print(self.mode)
            rect = camrectify(origin)
            # self.out.write(rect)
            # rect = cv2.flip(rect,0)
            # rect = cv2.flip(rect,1)
            warp = warp_img(rect)
            thresh = get_objectmask(warp)
            cv2.imshow('thresh', thresh)
            self.image = warp.copy()
            draw_img1 = warp.copy()
            self.get_bound(draw_img1, thresh, visualization=True)
            cx, cy = None, None
            lx, rx = None, None

            # self.handls = []
            # hsv = cv2.cvtColor(warp.copy(),cv2.COLOR_BGR2HSV)
            # hand_mask = cv2.inRange(hsv, Hand_low, Hand_high)
            # hand_mask = cv2.dilate(hand_mask, kernel = np.ones((7,7),np.uint8))
            # (_,hand_contours, hand_hierarchy)=cv2.findContours(hand_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # for i , contour in enumerate(hand_contours):
            #     area = cv2.contourArea(contour)
            #     if area>600 and area < 100000 and hand_hierarchy[0, i, 3] == -1:					
            #         x,y,w,h = cv2.boundingRect(contour)
            #         self.handls.append([x, y, w, h])
            
            result = hand_tracking(warp_img(rect), cache(10), cache(10)).get_result()
            num_hand_view = len(result)
            # if num_hand_view == 0:
            #     self.tip_deque.clear()
            #     self.tip_deque1.clear()
            #     self.tip_deque2.clear()
            if num_hand_view == 0:
                if len(self.hand_mask) > 0 and self.after_trigger:
                    if color_flag is not None:
                        object_mask = get_objectmask(deepcopy(self.image))
                        if color_flag == "yellow":
                            color_mask = get_yellow_objectmask(deepcopy(self.image))
                        elif color_flag == "blue":
                            color_mask = get_blue_objectmask(deepcopy(self.image))
                        # elif color_flag == "green":
                        #     color_mask = get_green_objectmask(deepcopy(self.image))
                        mask = self.hand_mask[0]
                        for i in range(1, len(self.hand_mask), 1):
                            mask = cv2.bitwise_or(self.hand_mask[i],mask)                     
                        mask = cv2.bitwise_and(mask,color_mask)
                        temp_result = []
                        for cx, cy in self.surfacels:
                            if mask[cy, cx] == 255:
                                temp_result.append((cx, cy))
                    else:
                        object_mask = get_objectmask(deepcopy(self.image))
                        mask = self.hand_mask[0]
                        for i in range(1, len(self.hand_mask), 1):
                            mask = cv2.bitwise_or(self.hand_mask[i],mask)                     
                        mask = cv2.bitwise_and(mask,object_mask)
                        temp_result = []
                        for cx, cy in self.surfacels:
                            if mask[cy, cx] == 255:
                                temp_result.append((cx, cy))
                    '''
                    multihand
                    '''
                    self.draw = draw_img1
                    print("getting bitwise and when there is one finger after palm")
                    #print([temp_result, tips[0], center,3])
                    #self.hand_mask = []
                    #self.after_trigger = False
                    #netsend(temp_result, flag=1, need_unpack=True)
                    self.last_select = temp_result
                    self.mode = 3
                   # self.center = center
                    #return [temp_result, tips[0], center,3]
                else:
                    netsend([777,888], need_unpack=False, flag=-19)
            '''
            one hand in the view
            '''
            if num_hand_view == 1:
                center = result[0][0]
                tips = result[0][1]
                radius = result[0][2]
                box = result[0][3]
                fake_tip, fake_center = result[0][4]
                app = result[0][5]
                cv2.drawContours(draw_img1, [app],-1,(255, 0, 0),1)
                for k in range(len(tips)):
                    cv2.circle(draw_img1,tips[k],10,(255, 0, 0),2)
                    cv2.line(draw_img1,tips[k],center,(255, 0, 0),2)
                num_tips = len(tips)
                label = self.test(box, draw_img1)
                self.onehand_center = center
                #print(box)
                #label = -1
                #label = classifier(draw_img1,self.image, box)
                #self.tip_deque.appendleft(tips)
            # '''
            # one hand and one finger, flag == 1
            # '''
                
                    #rospy.loginfo("mask, trigger:{},{}".format(len(self.hand_mask), self.after_trigger))
                #if num_tips == 1 and len(self.boxls) > 0 and label == 1:
                if len(self.hand_mask) > 0 and self.after_trigger:
                    if color_flag is not None:
                        object_mask = get_objectmask(deepcopy(self.image))
                        if color_flag == "yellow":
                            color_mask = get_yellow_objectmask(deepcopy(self.image))
                            netsend([777,888], need_unpack=False, flag=-200)
                        elif color_flag == "blue":
                            color_mask = get_blue_objectmask(deepcopy(self.image))
                            netsend([777,888], need_unpack=False, flag=-100)
                        # elif color_flag == "green":
                        #     color_mask = get_green_objectmask(deepcopy(self.image))
                        mask = self.hand_mask[0]
                        for i in range(1, len(self.hand_mask), 1):
                            mask = cv2.bitwise_or(self.hand_mask[i],mask)                     
                        mask = cv2.bitwise_and(mask,color_mask)
                        temp_result = []
                        for cx, cy in self.surfacels:
                            if mask[cy, cx] == 255:
                                temp_result.append((cx, cy))
                    else:
                        object_mask = get_objectmask(deepcopy(self.image))
                        mask = self.hand_mask[0]
                        for i in range(1, len(self.hand_mask), 1):
                            mask = cv2.bitwise_or(self.hand_mask[i],mask)   
                        #print(mask.dtype, object_mask.dtype)                  
                        mask = cv2.bitwise_and(mask,object_mask)
                        temp_result = []
                        for cx, cy in self.surfacels:
                            if mask[cy, cx] == 255:
                                temp_result.append((cx, cy))
                    '''
                    multihand
                    '''
                    self.draw = draw_img1
                    print("getting bitwise and when there is one finger after palm")
                    if len(tips) == 0:
                        rospy.logwarn("no finger tips")
                    else:
                        #print([temp_result, tips[0], center,3])
                        #self.hand_mask = []
                        #self.after_trigger = False
                        self.last_select = temp_result
                        self.mode = 3
                        #self.center = center
                        return [temp_result, tips[0], center,3]

                if len(self.boxls) > 0 and num_tips == 1 and label != 4:        
                    if len(self.hand_mask) == 0 or not self.after_trigger:
                        #rospy.loginfo("single pointing")
                        #point = max(tips, key=lambda x: np.sqrt((x[0]- center[0])**2 + (x[1] - center[1])**2))
                        point = tips[0]
                        self.tip_deque.appendleft(point)
                        #
                        length_ls = []
                        for x, y, w, h in self.boxls:
                            length_ls.append((get_k_dis((point[0], point[1]), (center[0], center[1]), (x+w/2, y+h/2)), (x+w/2, y+h/2)))
                        length_ls = filter(lambda x: (point[1] - x[1][1]) * (point[1] - center[1]) <= 0, length_ls)
                        length_ls = filter(lambda x: x[1][1] - point[1] < 0, length_ls)
                        length_ls = filter(lambda x: x[0] < 15, length_ls)
                        if len(length_ls) > 0:
                            x,y = min(length_ls, key=lambda x: distant((x[1][0], x[1][1]), (point[0], point[1])))[1]
                            ind = test_insdie((x, y), self.boxls)
                            x, y, w, h = self.boxls[ind]
                            cx, cy = self.surfacels[ind]
                            cv2.rectangle(draw_img1,(x,y),(x+w,y+h),(0,0,255),2)
                            #cv2.circle(draw_img1, (cx, cy), 5, (0, 0, 255), -1)
                            #cv2.putText(draw_img1,"pointed",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
                            
                            '''
                            flag is 1
                            '''
                            if self.trigger:
                                self.pick_tip = tuple([point[0],point[1]])
                            self.draw = draw_img1
                            self.last_select = [(cx, cy)]
                            netsend([cx, cy], need_unpack=False)
                            self.mode = 1
                            self.pick_center = center
                            return [[point[0],point[1]],(cx, cy), center,1]
                        else:
                            self.draw = draw_img1
                            self.mode = 1
                            self.pick_center = center
                            return [[point[0],point[1]], center,1]
            #  '''
            # one hand and two finger, flag == 2
            # '''
                elif num_tips == 2 and len(self.boxls) > 0 and label != 4:
                    boxls = deepcopy(self.boxls)
                    length_lsr = []
                    length_lsl = []
                    rpoint, lpoint = tips
                    for x, y, w, h in self.boxls:
                        length_lsr.append((get_k_dis((rpoint[0], rpoint[1]), (center[0], center[1]), (x+w/2, y+h/2)), (x+w/2, y+h/2)))
                    length_lsr = filter(lambda x: (rpoint[1] - x[1][1]) * (rpoint[1] - center[1]) <= 0, length_lsr)
                    length_lsr = filter(lambda x: x[0] < 20, length_lsr)
                    if len(length_lsr) > 0:
                        rx,ry = min(length_lsr, key=lambda x: distant((x[1][0], x[1][1]), (rpoint[0], rpoint[1])))[1]
                        rind = test_insdie((rx, ry), self.boxls)
                        rx, ry = self.surfacels[rind]
                        x, y, w, h = self.boxls[rind]
                        #rx, ry = int(x+w/2), int(y+h/2)
                        del boxls[rind]
                        cv2.rectangle(draw_img1,(x,y),(x+w,y+h),(0,0,255),2)
                        #cv2.putText(draw_img1,"pointed_right",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
                        if len(boxls) > 0:
                            for x, y, w, h in boxls:
                                length_lsl.append((get_k_dis((lpoint[0], lpoint[1]), (center[0], center[1]), (x+w/2, y+h/2)), (x+w/2, y+h/2)))
                            length_lsl = filter(lambda x: (lpoint[1] - x[1][1]) * (lpoint[1] - center[1]) <= 0, length_lsl)
                            length_lsl = filter(lambda x: x[0] < 20, length_lsl)
                            if len(length_lsl) > 0:
                                lx,ly = min(length_lsl, key=lambda x: distant((x[1][0], x[1][1]), (lpoint[0], lpoint[1])))[1]
                                lind = test_insdie((lx, ly), boxls)
                                lx, ly = self.surfacels[lind]
                                x, y, w, h = boxls[lind]
                                #lx, ly = int(x+w/2), int(y+h/2)
                                cv2.rectangle(draw_img1,(x,y),(x+w,y+h),(0,0,255),2)
                                #cv2.putText(draw_img1,"pointed_left",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
                                '''
                                flag is 2
                                '''
                                self.draw = draw_img1
                                self.last_select = [[rx, ry], [lx, ly]]
                                netsend([[rx, ry], [lx, ly]])
                                self.mode = 2
                                #self.center = center
                                self.pick_center = center
                                return [[tips[0][0], tips[0][1]], [tips[1][0], tips[1][1]], [rx, ry], [lx, ly], center,2]

                # '''
                # one hand and multi finger, flag == 3
                # '''
                elif num_tips > 0 and label == 3:
                    temp_center = (center[0], center[1] - 30)
                    if not self.trigger:
                        netsend(list(temp_center), need_unpack=False, flag=-18)
                    elif self.trigger:
                        # surface = np.ones(self.image.shape)
                        # cv2.circle(surface, center, 120, (255, 255, 255), -1)
                        # grayscaled = cv2.cvtColor(surface,cv2.COLOR_BGR2GRAY)
                        # retval, threshold = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
                        # self.hand_mask.append(threshold)
                        self.hand_mask = []
                        self.hand_mask.append(get_handmask(deepcopy(self.image), center))
                        rospy.loginfo("get brushed")
                        self.draw = draw_img1
                        self.trigger = False
                        self.mode = 3
                        rospy.loginfo("send center information :{}".format(list(temp_center)))
                        netsend(list(temp_center), need_unpack=False, flag=-8)
                        self.pick_center = center
                        #self.center = center
                        return [temp_center,3]

                elif label == 4 and len(self.boxls) > 0 and len(tips) > 0 and len(tips) < 4:
                    #point = max(tips, key=lambda x: np.sqrt((x[0]- center[0])**2 + (x[1] - center[1])**2))
                    point = fake_tip
                    center = fake_center
                    length_ls = []
                    for x, y, w, h in self.boxls:
                        length_ls.append((get_k_dis((point[0], point[1]), (center[0], center[1]), (x+w/2, y+h/2)), (x+w/2, y+h/2)))
                    #length_ls = filter(lambda x: (point[1] - x[1][1]) * (point[1] - center[1]) <= 0, length_ls)
                    #length_ls = filter(lambda x: (point[0] - x[1][0]) * (center[0] - x[1][0]) > 0, length_ls)
                    length_ls = filter(lambda x: x[1][1] - point[1] < 0, length_ls)
                    #print("haha", len(length_ls))
                    length_ls = filter(lambda x: x[0] < 50, length_ls)
                    #print("ddd", len(length_ls))
                    sub_result = []
                    if color_flag is not None:
                        object_mask = get_objectmask(deepcopy(self.image))
                        if color_flag == "yellow":
                            color_mask = get_yellow_objectmask(deepcopy(self.image))
                        elif color_flag == "blue":
                            color_mask = get_blue_objectmask(deepcopy(self.image))
                        if len(length_ls) > 0:
                            for i in range(len(length_ls)):
                                # x,y = min(length_ls, key=lambda x: distant((x[1][0], x[1][1]), (point[0], point[1])))[1]
                                # ind = test_insdie((x, y), self.boxls)
                                x,y = length_ls[i][1]
                                ind = test_insdie((x, y), self.boxls)
                                x, y, w, h = self.boxls[ind]
                                cx, cy = self.surfacels[ind]
                                if color_mask[cy, cx] == 255:
                                    sub_result.append((cx, cy))
                                cv2.rectangle(draw_img1,(x,y),(x+w,y+h),(0,0,255),2)
                                #cv2.circle(draw_img1, (cx, cy), 5, (0, 0, 255), -1)
                                #cv2.putText(draw_img1,"general",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
                            
                            '''
                            flag is 1
                            '''
                            self.draw = draw_img1
                            self.last_select = sub_result
                            self.mode = 6
                            self.pick_center = center
                            #self.center = center
                            return [sub_result, center ,6]
                        else:
                            self.draw = draw_img1
                            return None
                    
                    else:
                        if len(length_ls) > 0:
                            for i in range(len(length_ls)):
                            # x,y = min(length_ls, key=lambda x: distant((x[1][0], x[1][1]), (point[0], point[1])))[1]
                            # ind = test_insdie((x, y), self.boxls)
                                x,y = length_ls[i][1]
                                ind = test_insdie((x, y), self.boxls)
                                x, y, w, h = self.boxls[ind]
                                cx, cy = self.surfacels[ind]
                                sub_result.append((cx, cy))
                                cv2.rectangle(draw_img1,(x,y),(x+w,y+h),(0,0,255),2)
                                #cv2.circle(draw_img1, (cx, cy), 5, (0, 0, 255), -1)
                                #cv2.putText(draw_img1,"general",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
                            netsend(sub_result, need_unpack=True)
                            self.draw = draw_img1
                            self.last_select = sub_result
                            self.mode = 6
                            self.pick_center = center
                            #self.center = center
                            return [sub_result, center ,6]
                        else:
                            self.draw = draw_img1
                            return None
                    







            '''
            two hand in the view
            '''
            if num_hand_view == 2:
                lcenter = result[0][0]
                ltips = result[0][1]
                lnum_tips = len(ltips)
                lradius = result[0][2]
                lbox = result[0][3]
                llabel = self.test(lbox, draw_img1) 
                app = result[0][5]
                cv2.drawContours(draw_img1, [app],-1,(255, 0, 0),1)
                for k in range(len(ltips)):
                    cv2.circle(draw_img1,ltips[k],10,(255, 0, 0),2)
                    cv2.line(draw_img1,ltips[k],lcenter,(255, 0, 0),2)

                rcenter = result[1][0]
                rtips = result[1][1]
                rnum_tips = len(rtips)
                rradius = result[1][2]
                rbox = result[1][3]
                rlabel = self.test(rbox, draw_img1)
                lapp = result[1][5]
                cv2.drawContours(draw_img1, [lapp],-1,(255, 0, 0),1)
                for k in range(len(rtips)):
                    cv2.circle(draw_img1,rtips[k],10,(255, 0, 0),2)
                    cv2.line(draw_img1,rtips[k],rcenter,(255, 0, 0),2)
                # '''
                # two hand is both one finger pointing, ONLY PLACE
                # '''
                if set([lnum_tips, rnum_tips]) == set([1,1]) and len(self.boxls) > 0 and set([llabel, rlabel]) == set([1,1]):
                    self.draw = draw_img1
                    
                    '''
                    flag is 4
                    '''
                    self.mode = 4
                    self.two_hand_mode =4
                    self.tip_deque1.appendleft((ltips[0][0], ltips[0][1]))
                    self.tip_deque2.appendleft((rtips[0][0], rtips[0][1]))
                    self.center = [list(lcenter), list(rcenter)]
                    return [[rtips[0][0], rtips[0][1]], [ltips[0][0], ltips[0][1]], [list(rcenter), list(lcenter)], 4]

                elif max(set([lnum_tips, rnum_tips])) >= 2 and min(set([lnum_tips, rnum_tips])) == 1 and max(set([llabel, rlabel])) < 4 and self.onehand_center:
                    #sub_result = filter(lambda x: len(x[1]) == 1 , [[rcenter, rtips], [lcenter, ltips]])
                    sub_result = max([[rcenter, rtips], [lcenter, ltips]], key=lambda x: distant(x[0], self.onehand_center))
                    center = sub_result[0]
                    tips = sub_result[1]
                    # center = sub_result[0][0]
                    # tips = sub_result[0][1]
                    self.tip_deque.appendleft((tips[0][0], tips[0][1]))
                    self.draw = draw_img1
                    
                    if max(set([lnum_tips, rnum_tips])) == 2 and set([lnum_tips, rnum_tips]) == set([1,2]):
                        self.mode = 1
                        self.two_hand_mode = 1
                        return [[tips[0][0], tips[0][1]], 1]
                    else:
                        self.mode = 5
                        self.two_hand_mode = 5
                        return [[tips[0][0], tips[0][1]], 5]
                
                elif min(set([lnum_tips, rnum_tips])) == 1 and max(set([llabel, rlabel])) == 4 and self.onehand_center:
                    #sub_result = filter(lambda x: len(x[1]) == 1 , [[rcenter, rtips], [lcenter, ltips]])
                    sub_result = max([[rcenter, rtips], [lcenter, ltips]], key=lambda x: distant(x[0], self.onehand_center))
                    center = sub_result[0]
                    tips = sub_result[1]
                    # center = sub_result[0][0]
                    # tips = sub_result[0][1]
                    self.tip_deque.appendleft((tips[0][0], tips[0][1]))
                    self.draw = draw_img1
                    self.mode = 1
                    self.two_hand_mode = 1
                    #rospy.loginfo("jdjdjdjjs")
                    return [[tips[0][0], tips[0][1]], 1]
        self.draw = draw_img1       

    def get_bound(self, image, object_mask, visualization=True):
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
            #print(self.surfacels)

        # for x, y, w, h in self.boxls:
        #     sub = image[y:y+h, x:x+w, :]
        #     hsv = cv2.cvtColor(sub,cv2.COLOR_BGR2HSV)
        #     top_mask = cv2.inRange(hsv, Top_low, Top_high)
        #     (_,top_contours, object_hierarchy)=cv2.findContours(top_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #     max_area = 0
        #     for i , contour in enumerate(top_contours):
        #         area = cv2.contourArea(contour)
        #         if area>max_area and object_hierarchy[0, i, 3] == -1:					
        #             M = cv2.moments(contour)
        #             cx = int(M['m10']/M['m00'])
        #             cy = int(M['m01']/M['m00'])
        #             max_area = area
        #     self.surfacels.append((cx+x, cy+y))

    def reset(self): 
        self.tip_deque.clear()
        self.tip_deque1.clear()
        self.tip_deque2.clear()
        self.two_hand_mode = None
        self.mode = 0
        self.last_select = None
        self.center = None
        self.pick_tip = None
        self.pick_center = None
        self.hand_mask = []
        self.onehand_center = None

    def __del__(self):
        self.cap.release()
        self.out.release()

def callback(msg):
    #print(msg.data)
    global voice_flag
    voice_flag = msg.data

def colorback(msg):
    #print(msg.data)
    global color_flag
    color_flag = msg.data

if __name__ == '__main__':
    temp = temp_tracking()
    rospy.init_node('hand_tracking_node')
    pub = rospy.Publisher('/target_position', Int32MultiArray, queue_size=1)
    sub = rospy.Subscriber('/voice_command', Int32, callback)
    sub_for_color = rospy.Subscriber('/item_color', String, colorback)
    ready_for_place = False
    pos_N_cmd = []
    pick_handcenter = None
    valid = None
    pre_length = 0
    while True:
        
        
        #if pos:
            #=================PICK==============#
        if voice_flag == 9 or voice_flag == -20:
            ready_for_place = False
            pos_N_cmd = []
            color_flag = None
            voice_flag = None
            temp.reset()
            # temp.tip_deque.clear()
            # temp.tip_deque1.clear()
            # temp.tip_deque2.clear()
            # temp.two_hand_mode = None
            # temp.mode = 0
            # temp.last_select = None
            # temp.center = None
            # temp.pick_tip = None
            # temp.pick_center = None

            netsend([777, 888], need_unpack=False,flag=0)
            # temp.hand_mask = []
            # temp.onehand_center = None
        # if voice_flag == -9:
        #     netsend([777, 888], need_unpack=False,flag=-99)
        if voice_flag == 1 and not ready_for_place:
            temp.trigger = True
            pos = temp.update()
            
            print(temp.last_select)
            if True:
                if temp.mode == 1 and temp.last_select:
                    rospy.loginfo("one finger one hand")
                    pos_N_cmd.append(int(temp.last_select[0][0]))
                    pos_N_cmd.append(int(temp.last_select[0][1]))
                    netsend(pos_N_cmd, flag=1, need_unpack=False)
                elif temp.mode == 2 and temp.last_select:
                    rospy.loginfo("one hand two finger")
                    pos_N_cmd.append(int(temp.last_select[0][0]))
                    pos_N_cmd.append(int(temp.last_select[0][1]))
                    pos_N_cmd.append(int(temp.last_select[1][0]))
                    pos_N_cmd.append(int(temp.last_select[1][1]))
                    netsend(pos_N_cmd, flag=1, need_unpack=False)
                elif temp.mode == 3:
                    # start = time.time()
                    # while time.time() - start < 0.5:
                    #     temp.update()
                    #     rospy.loginfo("brushing")
                    rospy.loginfo("finish brushing")
                    rospy.loginfo("multifinger one hand previous")
                    rospy.loginfo('not getting new hand mask')
                elif temp.mode == 6 and temp.last_select:
                    rospy.loginfo("general pointing")
                    for cx, cy in temp.last_select:
                        pos_N_cmd.append(int(cx))
                        pos_N_cmd.append(int(cy))
                    netsend(pos_N_cmd, flag=1, need_unpack=False)
                pick_handcenter = temp.pick_center
                pick_tip = temp.pick_tip
                temp.trigger = False
                ready_for_place = True
                pre_length = len(pos_N_cmd)

                temp.tip_deque.clear()
                temp.tip_deque1.clear()
                temp.tip_deque2.clear()
            #==================PLACE====================#
        elif voice_flag == 2 and ready_for_place:
            rospy.loginfo("let's go")          
            temp.after_trigger = True
            pos = temp.update()
            #print(temp.tip_deque, "deque1111")
            c = Counter(tuple(temp.tip_deque))
            point1 = c.most_common(3)
            point = []
            if pick_tip and temp.two_hand_mode is None:
                for i in range(len(point1)):
                    if distant(point1[i][0],pick_tip) > 10:
                        rospy.loginfo("append a tip")                   
                        point.append(point1[i])
            else:
                rospy.loginfo("not append a tip")
                point = point1
            # if temp.pick_tip and distant(point[0][0],pick_tip) < 10:
            #     del point[0]
            #point = c.most_common(3)
            if len(point) == 0 or point[0][1] < 1:
                print(point)
                rospy.logwarn("not enoght deque point")
                rospy.loginfo("current pick_tip: {}".format(pick_tip))
                rospy.loginfo("current is None")
                rospy.loginfo("current mode: {}".format(temp.mode))
                rospy.loginfo("current two hand mode: {}".format(temp.two_hand_mode))
                ready_for_place = False
                pos_N_cmd = []
                color_flag = None
                voice_flag = None
                temp.reset()
                netsend([777, 888], need_unpack=False,flag=0)
            if len(point) > 0:
                rospy.loginfo("current pick_tip: {}".format(pick_tip))
                rospy.loginfo("current is None")
                rospy.loginfo("current mode: {}".format(temp.mode))
                rospy.loginfo("current two hand mode: {}".format(temp.two_hand_mode))
                if temp.mode == 1 and temp.two_hand_mode != 4:
                    rospy.loginfo("one finger one hand place")
                    #point_1 = []
                    # if pick_tip:
                    #     for i in range(len(point)):
                    #         if distant(point[i][0],pick_tip) > 10:
                    #             rospy.loginfo("append a tip")                   
                    #             point_1.append(point[i])
                    #print(point_1)
                    # if len(point_1) > 0 and point_1[0][1] > 1:
                    #     pos_N_cmd.append(int(point_1[0][0][0]))
                    #     pos_N_cmd.append(int(point_1[0][0][1]))
                    if len(point) > 0 and point[0][1] > 1:
                        pos_N_cmd.append(int(point[0][0][0]))
                        pos_N_cmd.append(int(point[0][0][1]))
                    else:
                        rospy.logwarn("fail to get the one finger place: {}".format(point))
                elif temp.mode == 3 and len(pos_N_cmd) == 0:
                    rospy.loginfo("multifinger one hand place")
                    for cx, cy in temp.last_select:
                        pos_N_cmd.append(cx)
                        pos_N_cmd.append(cy)
                    netsend(pos_N_cmd, flag=1, need_unpack=False)
                    rospy.sleep(0.1)
                    print(point)
                    if len(point) > 0 and point[0][1] > 1:
                        pos_N_cmd.append(int(point[0][0][0]))
                        pos_N_cmd.append(int(point[0][0][1]))
                    else:
                        ready_for_place = False
                        pos_N_cmd = []
                        color_flag = None
                        voice_flag = None
                        temp.reset()
                        netsend([777, 888], need_unpack=False,flag=0)
                    temp.hand_mask = []
                    #temp.trigger = True
                

                elif temp.mode == 5 and len(temp.hand_mask) > 0:
                    rospy.loginfo("two hand one multi one finger place")
                    if point[0][1] > 1:
                        netsend([int(point[0][0][0]),int(point[0][0][1])], flag=-10, need_unpack=False)
                    rospy.sleep(5)
                    image = temp.get_current_frame()
                    object_mask = get_objectmask(image)
                    if color_flag is not None:
                        object_mask = get_objectmask(deepcopy(image))
                        if color_flag == "yellow":
                            color_mask = get_yellow_objectmask(deepcopy(image))
                        elif color_flag == "blue":
                            color_mask = get_blue_objectmask(deepcopy(image))
                        # elif color_flag == "green":
                        #     color_mask = get_green_objectmask(deepcopy(self.image))
                        mask = temp.hand_mask[0]
                        for i in range(1, len(temp.hand_mask), 1):
                            mask = cv2.bitwise_or(temp.hand_mask[i],mask)                     
                        mask = cv2.bitwise_and(mask,color_mask)
                        #mask = cv2.bitwise_and(temp.hand_mask,color_mask)
                    else:
                        mask = temp.hand_mask[0]
                        for i in range(1, len(temp.hand_mask), 1):
                            mask = cv2.bitwise_or(temp.hand_mask[i],mask)                     
                        mask = cv2.bitwise_and(mask,object_mask)
                    temp.get_bound(image, object_mask)


                    print(temp.surfacels)
                    for cx, cy in temp.surfacels:
                        if mask[cy, cx] == 255:
                            pos_N_cmd.append(int(cx))
                            pos_N_cmd.append(int(cy))
                    netsend(pos_N_cmd, flag=1, need_unpack=False)
                    rospy.sleep(0.1)
                    if point[0][1] > 1:
                        pos_N_cmd.append(int(point[0][0][0]))
                        pos_N_cmd.append(int(point[0][0][1]))
                    else:
                        ready_for_place = False
                        pos_N_cmd = []
                        color_flag = None
                        voice_flag = None
                        temp.reset()
                        netsend([777, 888], need_unpack=False,flag=0)
                    print("getting bitwise for two hand")
                    temp.hand_mask = []
                
                elif temp.mode == 4 or temp.two_hand_mode == 4:
                    rospy.loginfo("two hand one point place")
                    print("center", temp.center)
                    print("previous center", pick_handcenter)
                    if pos:
                        try:
                            ind = pos[-2].index(max(pos[-2], key=lambda x: sqrt((x[0] - pick_handcenter[0])**2 + (x[1] - pick_handcenter[1])**2)))
                            pos_N_cmd.append(int(pos[ind][0]))
                            pos_N_cmd.append(int(pos[ind][1]))
                            rospy.loginfo("it is in the view")
                        except:
                            ready_for_place = False
                            pos_N_cmd = []
                            color_flag = None
                            voice_flag = None
                            temp.reset()
                            netsend([777, 888], need_unpack=False,flag=0)
                    elif len(point) > 0:
                        rospy.loginfo("it is not in the view")
                        ind = temp.center.index(max(temp.center, key=lambda x: sqrt((x[0] - pick_handcenter[0])**2 + (x[1] - pick_handcenter[1])**2)))
                        if ind == 0:
                            c = Counter(list(temp.tip_deque1))
                            point = c.most_common(1)
                        elif ind == 1:
                            c = Counter(list(temp.tip_deque2))
                            point = c.most_common(1)
                        if len(point) > 0 and point[0][1] > 1:
                            pos_N_cmd.append(int(point[0][0][0]))
                            pos_N_cmd.append(int(point[0][0][1]))
                pick_handcenter = None
                temp.after_trigger = False
                #netsend(pos_N_cmd)
                if len(pos_N_cmd) != pre_length and len(pos_N_cmd) >= 4:
                    netsend(pos_N_cmd[-2:], flag=2, need_unpack=False)
                    print(pos_N_cmd, "this is published msg")
                    pub.publish(Int32MultiArray(data=pos_N_cmd))
                else:
                    netsend([777, 888], need_unpack=False,flag=0)
                    rospy.logwarn("fail to publish, need to have place location")
                    ready_for_place = False
                    pos_N_cmd = []
                    color_flag = None
                    voice_flag = None
                    temp.reset()
                    #netsend([777, 888], need_unpack=False,flag=0)
                voice_flag = None
                temp.reset()
                ready_for_place = False
                color_flag = None
                pos_N_cmd = []
            
        elif voice_flag == -1:
            break
        else:
            pos = temp.update()
            #netsend(temp.last_select)
        temp.out.write(temp.draw)
        cv2.imshow('node',temp.draw)
        k = cv2.waitKey(1) & 0xFF # large wait time to remove freezing
        if k == 113 or k == 27:
            break
    cv2.destroyAllWindows()

