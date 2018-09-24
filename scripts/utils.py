import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import random
import math
import cv2
from constant import *

AUGMENT = True
BATCH_SIZE = 4
output_vector = [10, 6, 1]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(10368, 200)
        self.fc2 = nn.Linear(200, 4)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.batchnorm2(self.conv2(x)), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.batchnorm4(self.conv4(x)), 2))
        x = x.view(-1, 10368)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



class CovnetDataset(Dataset):
    def __init__(self, reader, transforms=None):
        self.reader = reader
        self.transform = transforms
    def __getitem__(self, item):
        image_tuple = self.reader[item]
        img1_dir = image_tuple[0]
        img1 = Image.open(img1_dir)
        label = float(image_tuple[1])
        result = torch.from_numpy(np.array([label], dtype=float))

        if AUGMENT:
            rotate_range = random.uniform(-180, 180)
            translation_range = random.uniform(-10, 10)
            scale_range = random.uniform(0.7, 1.3)
            if np.random.random() < 0.7:
                img1 = img1.rotate(rotate_range)
            if np.random.random() < 0.7:
                 img1 = img1.transform((img1.size[0], img1.size[1]), Image.AFFINE, (1, 0, translation_range, 0, 1, translation_range))
            if np.random.random() < 0.5:
                img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() < 0.5:
                img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)

        img1 = self.transform(img1)
        return (img1, result)
    def __len__(self):
        return len(self.reader)

def side_finder(frame, color):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    red = [Red_low, Red_high]
    blue = [Blue_low, Blue_high]
    if color == 'red':
        mask = cv2.inRange(hsv, *red)
    elif color == 'blue':
        mask = cv2.inRange(hsv, *blue)
    else:
        raise NameError('red or blue')
    kernal = np.ones((3 ,3), "uint8")
    mask = cv2.dilate(mask, kernal)   
    _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    center_list = []
    for i , contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100 and hierarchy[0, i, 3] == -1:					
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # cv2.circle(frame, (cx, cy), 10, [0, 255, 0])
            center_list.append((cx, cy))
    return center_list

def test_insdie(point, boundingbox_list):
    cx, cy = point
    for i, boundingbox in enumerate(boundingbox_list):
        x,y,w,h = boundingbox
        if cx > x and cx < x+w and cy > y and cy < y+h:
            return i


class cache():
    def __init__(self, length):
        self.list = []
        self.length = length
        self.full = False
    
    def append(self, data):
        if len(self.list) < self.length:
            self.list.append(data)
        else:
            del self.list[0]
            self.append(data)
            self.full = True

    def clear(self):
        self.list = []
        self.full = False



