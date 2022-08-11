import time

import rospy
from move_base_msgs.msg import MoveBaseActionGoal

if __name__ == '__main__':
    rospy.init_node("test_send_goal_node", anonymous=True)
    pub = rospy.Publisher('move_base/goal', MoveBaseActionGoal, queue_size=1)
    goal_msg = MoveBaseActionGoal()
    goal_msg.goal_id = time.time()
    goal_msg.goal.target_pose.pose.position.x = -0.0201952750817
    goal_msg.goal.target_pose.pose.position.y = -2.51371260114
    goal_msg.goal.target_pose.pose.position.z = 0.138
    goal_msg.goal.target_pose.pose.orientation.x = 0.0
    goal_msg.goal.target_pose.pose.orientation.y = 0.0
    goal_msg.goal.target_pose.pose.orientation.z = 0.0391967644333
    goal_msg.goal.target_pose.pose.orientation.w = 0.999231511542
    pub.publish(goal_msg)
