from __future__ import print_function
import cv2
import numpy as np
import heapq
from utils import cache
from constant import Hand_low, Hand_high
from utils import cache
import math

distant = lambda (x1, y1), (x2, y2) : math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
Finger_distanct = 20

class hand_tracking():
    def __init__(self, image, memory1, memory2):
        self.memory1 = memory1
        self.memory2 = memory2
        self.flag = False
        self.cnt_pts = []
        frame = image.copy()
        self.radius_thresh = 0.05
        self.result = []
        #_, frame = cap.read()
        #frame = self.warp(frame)
        blur = cv2.blur(frame,(5,5))
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        kernal = np.ones((7 ,7), "uint8")
        mask = cv2.inRange(hsv, Hand_low, Hand_high)
        #mask = cv2.dilate(mask, kernal)
        mask2 = cv2.GaussianBlur(mask,(11,11),-1)  
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

        dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)

        median = cv2.medianBlur(dilation2,5)
        _,thresh = cv2.threshold(median,127,255,0)
        # cv2.imshow('thresh', thresh)
        
        # cv2.imshow('dgs', mask)
        self.mask = mask.copy()
        _, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        self.hand_cnt = [] 
        self.only_point = None
        self.rl_point = None
        self.center = None
        self.angle = None
        self.twoangle = None
        self.cnt = None
        max_area = 1000
        # try:	
        for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if area>2000 and area < 10000:
                cnts = contours[i]
                
                epsilon = 0.001*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                self.cnt = approx
                hull = cv2.convexHull(cnts)
                frame,hand_center,hand_radius = self.mark_hand_center(frame, cnts)
                
                frame,finger=self.mark_fingers(frame,hull,hand_center,hand_radius)

                cv2.drawContours(frame,[approx],-1,(0, 0, 255),1)

                
        cv2.imshow('hand_tracking',frame) 
    def get_result(self):
        #self.filter()
        # return (self.only_point, self.angle), (self.rl_point, self.twoangle), self.center
        if len(self.result) == 2:
            if self.result[0][0][0] > self.result[1][0][0]:
                #self.result = [self.result[1], self.result[0]]
                self.result = self.result[::-1]
        return self.result
    
    def filter(self):
        if self.memory1.full:
            if 0 in self.memory1.list:
                self.only_point = None
            self.memory1.clear()
        else:
            self.only_point = None

        if self.memory2.full:
            if 0 in self.memory2.list:
                self.rl_point = None
            self.memory2.clear()
        else:
            self.rl_point = None
        # self.memory.append()

    def mark_hand_center(self, frame_in,cont):    
        max_d=0
        pt=(0,0)
        x,y,w,h = cv2.boundingRect(cont)
        self.box = (x,y,w,h)
        for ind_y in xrange(int(y),int(y+h)): 
            for ind_x in xrange(int(x),int(x+w)): 
                dist= cv2.pointPolygonTest(cont,(ind_x,ind_y),True)
                if(dist>max_d):
                    max_d=dist
                    pt=(ind_x,ind_y)
        cv2.circle(frame_in,pt,int(max_d),(0,0,255),2)
        sub_thresh = self.mask[y:y+h, x:x+w].copy()
        mat = np.argwhere(sub_thresh != 0)
        mat[:, [0, 1]] = mat[:, [1, 0]]
        mat = np.array(mat).astype(np.float32) #have to convert type for PCA
        m, e = cv2.PCACompute(mat, mean = np.array([]))
        center = tuple(m[0])
        center = tuple([pt[0], pt[1]])
        rows,cols = frame_in.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(cont, cv2.DIST_L2,0,0.01,0.01)
        cv2.line(frame_in,(x - vx * 70, y - vy * 70),(x,y),(0,255,0),2)
        endpoint1 = (x - vx * 70, y - vy * 70)
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((cols-x)*vy/vx)+y)
        
        #cv2.line(frame_in,(cols-1,righty),(0,lefty),(0,255,0),2)
        # ec = cv2.fitEllipse(cont)
        # (x,y),(MA,ma),angle = ec
        # cv2.ellipse(frame_in,ec,(0, 0, 255),1)
        # print(angle)
        # if angle != 0:
        #     if angle > 90:
        #         k = math.tan(math.radians(90 + angle))
        #         x1 = center[0] - 1/k
        #         y1 = center[1] - 1
        #     elif angle>0 and angle < 90:
        #         #print("hg")
        #         k = math.tan(math.radians(90 -angle))
        #         x1 = center[0] - 1/k
        #         y1 = center[1] - 1
        # else:
        #     x1 = center[0]
        #     y1 = center[1] - 10
        #endpoint1 = tuple([x1, y1])
        #endpoint1 = tuple(m[0] + e[0]*10)
        #endpoint1 = tuple(m[0] + k*10)

        if endpoint1[0] < x:
            self.end = (endpoint1,(x, y))
        else:
            self.end = ((x, y), endpoint1)
        #self.end = ((int(endpoint1[0] + x), int(endpoint1[1] + y)), (int(center[0] + x), int(center[1] + y)))
        #cv2.circle(frame_in,self.end[0],5,(0,255,255),-1)
        #cv2.circle(frame_in,self.end[1],5,(0,255,255),-1)
        #print(len(cont))
        for [cnt_pts] in cont:
            self.cnt_pts.append(distant(cnt_pts, pt))
        #print(self.cnt_pts)
        return frame_in,pt,max_d

    def mark_fingers(self, frame_in,hull,pt,radius):
      
        finger=[(hull[0][0][0],hull[0][0][1])]
        j=0

        cx = pt[0]
        cy = pt[1]
        self.center = (cx, cy)
        for i in range(len(hull)):
            dist = np.sqrt((hull[-i][0][0] - hull[-i+1][0][0])**2 + (hull[-i][0][1] - hull[-i+1][0][1])**2)
            if dist>Finger_distanct:
                if(j==0):
                    finger=[(hull[-i][0][0],hull[-i][0][1])]
                else:
                    finger.append((hull[-i][0][0],hull[-i][0][1]))
                j=j+1

        finger = filter(lambda x: x[1] < cy, finger)
        finger = filter(lambda x: np.sqrt((x[0]- cx)**2 + (x[1] - cy)**2) > 1.7 * radius, finger)
        self.result.append([(cx, cy), finger, radius, self.box, self.end, self.cnt])
    

        for k in range(len(finger)):
            cv2.circle(frame_in,finger[k],10,(0, 0, 255),2)
            cv2.line(frame_in,finger[k],(cx,cy),(0, 0,255),2)
        return frame_in,finger
def warp(img):
    #pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
    pts1 = np.float32([[206,138],[577,114],[208,355],[596,347]])
    pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(640,480))
    return dst
    
        




if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        OK, origin = cap.read()
        if OK:
            ob = hand_tracking(warp(origin), cache(10), cache(10))
            #print(ob.get_result())
        # if ob.angle is not None:
        #     print(ob.angle)
        k = cv2.waitKey(1) & 0xFF # large wait time to remove freezing
        if k == 113 or k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
