# Contains the timer of the robot

import rospy
from std_msgs.msg import Int8


class Timer():

    def __init__(self, time_limit):
        self.pub_timer = rospy.Publisher('timer', Int8, queue_size=0)
        self.timer = rospy.Timer(rospy.Duration(
            time_limit), self.publish, oneshot=True)

    def publish(self, event):
        """Publishes when the timer has ended."""

        self.pub_timer.publish(1)

    def shutdown(self):
        """Stops the timer."""
        self.timer.shutdown()
