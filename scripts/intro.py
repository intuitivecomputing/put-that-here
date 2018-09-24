from __future__ import print_function
import numpy as np
import time
import cv2
import rospy
from std_msgs.msg import Int32MultiArray

def coor_tran(pos):
    for i in range(int(len(pos) / 2)):
        pos[2 * i] = int( pos[2 * i] * 1600 / 640 )
        pos[2 * i + 1] = int( pos[2 * i + 1] * 900 / 480 )
    return pos

def draw_x(img, x, y, line):
    cv2.line(img, (x + int(line/2), y + int(line/2)),(x - int(line/2), y - int(line/2)), (255, 255, 255), 10)
    cv2.line(img, (x - int(line / 2), y + int(line / 2)), (x + int(line / 2), y - int(line / 2)), (255, 255, 255), 10)
    return img

def draw_arrow(img, x1, y1, x2, y2, line):
    cv2.arrowedLine(img, (int(0.6*x1 + 0.4*x2), int(0.6*y1 + 0.4*y2)), (int(0.4*x1 + 0.6*x2), int(0.4*y1+0.6*y2)),(255, 255, 255), line)
    return img


def white():
    img = np.ones((900, 1600, 3), np.uint8) * 255
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)


def intro():
    img = np.zeros((900, 1600, 3), np.uint8) * 255
    cv2.circle(img, (int(1600*428/640) -50,int(207*900/480) +35 ), 40, (255, 255, 255), -1)
    img = draw_x(img,int(325*1600/640),int(420*900/480), 40)
    img = draw_arrow(img, int(1600*428/640) - 20, int(207*900/480) + 35, int(1600*325/640),int(420*900/480),5)
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)


    img = np.zeros((900, 1600, 3), np.uint8) * 255
    cv2.rectangle(img, (102*1600/640, 32*900/480), (536*1600/640, 389*900/480), (255, 255, 255), 5)
    cv2.circle(img, (int(1600*428/640),int(207*900/480)), 40, (255, 255, 255), -1)
    cv2.circle(img, (int(1600*428/640+60),int(207*900/480+60)), 40, (255, 255, 255), -1)
    cv2.circle(img, (int(1600*428/640+120),int(207*900/480)), 40, (255, 255, 255), -1)
    img = draw_x(img,int(325*1600/640),int(420*900/480), 40)
    img = draw_arrow(img, int(1600*428/640+40), int(207*900/480+40), int(1600*325/640),int(420*900/480),5)
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)

    img = np.zeros((900, 1600, 3), np.uint8) * 255
    cv2.circle(img, (int(1600*428/640 +40),int(207*900/480 + 20)), 120, (255, 255, 255), 5)
    img = draw_x(img,int(325*1600/640),int(420*900/480), 40)
    img = draw_arrow(img, int(1600*428/640 + 20), int(207*900/480 + 20), int(1600*325/640),int(420*900/480),5)
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)

    img = np.zeros((900, 1600, 3), np.uint8) * 255
    cv2.circle(img, (int(1600*200/640),int(400*900/480)), 120, (255, 255, 255), 5)
    img = draw_x(img,int(325*1600/640),int(420*900/480), 40)
    img = draw_arrow(img, int(1600*200/640), int(400*900/480), int(1600*325/640),int(420*900/480),5)
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)

    img = np.zeros((900, 1600, 3), np.uint8) * 255
    cv2.circle(img, (int(1600*500/640),int(400*900/480)), 40, (255, 255, 255), -1)
    cv2.circle(img, (int(1600*500/640+80),int(400*900/480)), 40, (255, 255, 255), -1)
    img = draw_x(img,int(325*1600/640),int(420*900/480), 40)
    img = draw_arrow(img, int(1600*500/640+40), int(400*900/480), int(1600*325/640),int(420*900/480),5)
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)







    img = np.zeros((900, 1600, 3), np.uint8) * 255
    cv2.circle(img, (int(1600*500/640),int(400*900/480)), 40, (255, 255, 255), -1)
    img = draw_x(img,int(325*1600/640),int(420*900/480), 40)
    img = draw_arrow(img, int(1600*500/640), int(400*900/480), int(1600*325/640),int(420*900/480),5)
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)


    img = np.zeros((900, 1600, 3), np.uint8) * 255
    cv2.circle(img, (int(1600*200/640),int(400*900/480)), 40, (255, 255, 255), -1)
    cv2.circle(img, (int(1600*200/640+100),int(400*900/480)), 40, (255, 255, 255), -1)
    cv2.circle(img, (int(1600*200/640+50),int(400*900/480)-60), 40, (255, 255, 255), -1)
    img = draw_x(img,int(325*1600/640),int(420*900/480), 40)
    img = draw_arrow(img, int(1600*200/640+30), int(400*900/480+30), int(1600*325/640),int(420*900/480),5)
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)

    img = np.zeros((900, 1600, 3), np.uint8) * 255
    cv2.circle(img, (int(1600*275/640 +50),int(190*900/480)), 120, (255, 255, 255), 5)
    img = draw_x(img,int(325*1600/640),int(420*900/480), 40)
    img = draw_arrow(img, int(1600*275/640 + 50), int(190*900/480 + 20), int(1600*325/640),int(420*900/480),5)
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)

    img = np.zeros((900, 1600, 3), np.uint8) * 255
    cv2.circle(img, (int(1600*500/640 + 40),int(400*900/480)), 120, (255, 255, 255), 5)
    img = draw_x(img,int(325*1600/640),int(420*900/480), 40)
    img = draw_arrow(img, int(1600*500/640 + 40), int(400*900/480), int(1600*325/640),int(420*900/480),5)
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)

    img = np.zeros((900, 1600, 3), np.uint8) * 255
    cv2.circle(img, (int(1600*275/640),int(190*900/480)), 40, (255, 255, 255), -1)
    cv2.circle(img, (int(1600*275/640+80),int(190*900/480)), 40, (255, 255, 255), -1)
    img = draw_x(img,int(325*1600/640),int(420*900/480), 40)
    img = draw_arrow(img, int(1600*275/640+40), int(190*900/480), int(1600*325/640),int(420*900/480),5)
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)







    img = np.zeros((900, 1600, 3), np.uint8) * 255
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    intro()