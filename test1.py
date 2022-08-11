# coding=utf-8
import rospy
import time
from dobot.srv import SetHOMECmd				 #调用回零话题
from dobot.srv import SetEndEffectorSuctionCup   #调用末端回零话题
from dobot.srv import SetPTPCmd					#调用机械臂点到点运动话题

grap_up_x = 7
grap_up_y = -257
grap_up_z = 95
grap_up_r = -170

capture_pose_x = 0
capture_pose_y = 0

grap_up_x = 7
grap_up_y = -257
grap_up_z = 95
grap_up_r = -170

grap_down_x = 17
grap_down_y = -310
grap_down_z = -22
grap_down_r = -170

put_up_x = 7
put_up_y = 185
put_up_z = 67
put_up_r = 2

put_down_x = 7
put_down_y = 170
put_down_z = -29
put_down_r = 2

deliver_up_x = 7
deliver_up_y = -319
deliver_up_z = 25
deliver_up_r = -170

def init_arm():
    rospy.wait_for_service('DobotServer/SetHOMECmd')
    rospy.wait_for_service('DobotServer/SetPTPCmd')
    #t0 = rospy.Duration(5, 0)
    try:
        client = rospy.ServiceProxy('DobotServer/SetHOMECmd', SetHOMECmd)
        PTP_client = rospy.ServiceProxy('DobotServer/SetPTPCmd', SetPTPCmd)
        response = client()
        #rospy.sleep(t0)
	#response = PTP_client(1, grap_down_x, grap_down_y, grap_down_z, grap_down_r)
        response = PTP_client(1, grap_up_x, grap_up_y, grap_up_z, grap_up_r)  # houmian,shangfang
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

def arm_grap():
    global capture_pose_x, capture_pose_y, grap_down_x, grap_down_y
    rospy.wait_for_service('DobotServer/SetEndEffectorSuctionCup')
    rospy.wait_for_service('DobotServer/SetPTPCmd')
    t1 = rospy.Duration(1,0)
    try:
        end_client = rospy.ServiceProxy('DobotServer/SetEndEffectorSuctionCup', SetEndEffectorSuctionCup)
        PTP_client = rospy.ServiceProxy('DobotServer/SetPTPCmd', SetPTPCmd)
        response = PTP_client(1, grap_up_x, grap_up_y, grap_up_z, grap_up_r)  # houmian,shangfang
        response = PTP_client(1, grap_down_x, grap_down_y, grap_down_z, grap_down_r)  # houmian,fangxia
        response = end_client(1, 1, True)
        rospy.sleep(t1)
        response = PTP_client(1, grap_up_x, grap_up_y, grap_up_z, grap_up_r)  # houmian,shangfang
        response = PTP_client(1, put_up_x, put_up_y, put_up_z, put_up_r)  # qianmian,shangfang
        response = PTP_client(1, put_down_x, put_down_y, put_down_z, put_down_r)  # qianmian,fangxia
        response = end_client(0, 0, True)
        rospy.sleep(t1)
        response = PTP_client(1, put_up_x, put_up_y, put_up_z, put_up_r)  # qianmian,shangfang
        #print("grap_down_x", grap_down_x, "grap_down_y", grap_down_y)


    except rospy.ServiceException, e:
        print"Service call failed: %s" % e

def arm_deliver():
    rospy.wait_for_service('DobotServer/SetEndEffectorSuctionCup')
    rospy.wait_for_service('DobotServer/SetPTPCmd')
    t2 = rospy.Duration(1, 0)
    try:
        end_client = rospy.ServiceProxy('DobotServer/SetEndEffectorSuctionCup', SetEndEffectorSuctionCup)
        PTP_client = rospy.ServiceProxy('DobotServer/SetPTPCmd', SetPTPCmd)
        response = PTP_client(1, put_down_x, put_down_y, put_down_z, put_down_r)
        response = end_client(1, 1, True)
        rospy.sleep(t2)
        response = PTP_client(1, put_up_x, put_up_y, put_up_z, put_up_r)  # qianmian,shangfang
        response = PTP_client(1, deliver_up_x, deliver_up_y, deliver_up_z, deliver_up_r)
        response = end_client(0, 0, True)
        response = PTP_client(1, grap_up_x, grap_up_y, grap_up_z, grap_up_r)  # houmian,shangfang

        rospy.sleep(t2)
    except rospy.ServiceException, e:
        print"Service call failed: %s" % e


if __name__ == "__main__":
   init_arm()
   arm_grap()
   arm_deliver()
