#! /usr/bin/env python
# coding=utf-8

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def doMsg(msg):
    rospy.loginfo("I heard:%s", msg.data)

def callback(data):

    cv2.imwrite("/home/eaibot/cam_test.jpg", imgdata)

def listener():
    rospy.init_node('listener', anonymous=True)
    data = rospy.wait_for_message("/usb_cam/image_correct", Image, timeout=10)
    imgdata = CvBridge().imgmsg_to_cv2(data, "rgb8")

    cv2.flip(imgdata, 0, v_image)
    # rospy.Subscriber("/usb_cam/image_correct", Image, callback)

if __name__ == "__main__":
    # 2.初始化 ROS 节点:命名(唯一)
    rospy.init_node("listener_p")
    # 3.实例化 订阅者 对象
    sub = rospy.Subscriber("chatter", Image, doMsg, queue_size=10)
    # 4.处理订阅的消息(回调函数)
    # 5.设置循环调用回调函数
    rospy.spin()
