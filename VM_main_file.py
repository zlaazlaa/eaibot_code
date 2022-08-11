# coding=utf-8
import os
import time

import rospy

from std_msgs.msg import String


def start_ocr(data):
    if data == 'start_ocr':
        os.system('python /home/eaibot/aistudio/work/Ubuntu/predict_backup_2.py')  # 运行识别程序
        time.sleep(3)
        f = open("/home/eaibot/aistudio/work/Ubuntu/result.txt", "r")
        result_str = f.read()
        pub = rospy.Publisher('/ocr_result', String, queue_size=1)
        rate = rospy.Rate(100)  # 100hz
        pub.publish(result_str)
        rate.sleep()


if __name__ == '__main__':
    # 建立节点
    rospy.init_node('VM_main_node', anonymous=True)
    # 订阅话题

    rospy.Subscriber('/start_ocr', String, start_ocr)

    # 调用回调函数，并阻塞，直到程序结束
    rospy.spin()
