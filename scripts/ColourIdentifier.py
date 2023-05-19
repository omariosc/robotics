# Contains the class used to identify the colour of an object in an image.

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Int8, Int64
from cv_bridge import CvBridge, CvBridgeError


class ColourIdentifier():
    
    """
    Creates a colour identifier object.
    
    Args:
        debug (bool): Whether debug code should be executed
    
    Returns:
        ColourIdentifier: A colour identifier object
    """
    def __init__(self, debug=False):
        self.sensitivity = 10
        self.bridge = CvBridge()
        self.image = None

        # Subscribes to the image topic
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)

        # Publishes the image and the name of the object
        self.pub_image = rospy.Publisher(
            'identified_image', Image, queue_size=0)
        self.pub_text = rospy.Publisher('identified_name', Int8, queue_size=0)
        self.pub_size = rospy.Publisher('identified_size', Int64, queue_size=0)
        self.pub_center = rospy.Publisher(
            'identified_center', Int64, queue_size=0)

        # Will control whether debug code (logging, showing images etc.) will be executed.
        self.debug = debug

    def callback(self, data):
        """Displays the received image.

        Args:
            data (sensor_msgs.msg.Image): The received image
        
        Returns:
            None
        """
        # Convert the received image into a opencv image
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as error:
            print(error)

        # Convert the rgb image into a hsv image
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Lower and upper bounds for Yellow, Red, Blue, Purple.
        hsv_r_lower = np.array([0 - self.sensitivity, 90, 90])
        hsv_r_upper = np.array([0 + self.sensitivity, 180, 180])
        hsv_y_lower = np.array([30 - self.sensitivity, 50, 50])
        hsv_y_upper = np.array([50 + self.sensitivity, 160, 160])
        hsv_b_lower = np.array([100 - self.sensitivity, 100, 100])
        hsv_b_upper = np.array([130 + self.sensitivity, 255, 255])
        hsv_p_lower = np.array([140 - self.sensitivity, 20, 20])
        hsv_p_upper = np.array([160 + self.sensitivity, 100, 100])

        # Creating the masks:
        red_mask = cv2.inRange(hsv_image, hsv_r_lower, hsv_r_upper)
        yellow_mask = cv2.inRange(hsv_image, hsv_y_lower, hsv_y_upper)
        blue_mask = cv2.inRange(hsv_image, hsv_b_lower, hsv_b_upper)
        purple_mask = cv2.inRange(hsv_image, hsv_p_lower, hsv_p_upper)

        mask_ry = cv2.bitwise_or(red_mask, yellow_mask)
        mask_ryb = cv2.bitwise_or(mask_ry, blue_mask)
        mask_rybp = cv2.bitwise_or(mask_ryb, purple_mask)
        result = cv2.bitwise_and(self.image, self.image, mask=mask_rybp)

        self.scarlet(red_mask)
        self.mustard(yellow_mask)
        self.peacock(blue_mask)
        self.plum(purple_mask)

        if self.debug:
            # Show the resultant images you have created.
            cv2.namedWindow('[ColourIdentifier] Debug Output')
            cv2.imshow('[ColourIdentifier] Debug Output', result)
            cv2.waitKey(3)

    def publish(self, image, image_id, size, center):
        """
        Publishes the image and the name of the object.

        Args:
            image (numpy.ndarray): The OpenCV image of the object
            image_id (str): The name of the object using position in the list
            size (float): The size of the object
            center (tuple): The center of the object
        
        Returns:
            None
        """

        self.pub_image.publish(
            self.bridge.cv2_to_imgmsg(image, encoding="bgr8"))
        self.pub_text.publish(image_id)
        self.pub_size.publish(int(size))
        self.pub_center.publish(center)

    def mustard(self, maskimage):
        """
        Callback checking for Mustard.

        Args:
            maskimage (numpy.ndarray): Masked image of the object
        
        Returns:
            None
        """

        yellowcontours, _ = cv2.findContours(
            maskimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(yellowcontours) > 0:
            yellow_c = max(yellowcontours, key=cv2.contourArea)
            size = cv2.contourArea(yellow_c)
            if size > 800:
                self.publish(self.image, 0, size, self.get_center(yellow_c))

    def peacock(self, maskimage):
        """
        Callback checking for Peacock.

        Args:
            maskimage (numpy.ndarray): Masked image of the object
        
        Returns:
            None
        """

        bluecontours, _ = cv2.findContours(
            maskimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(bluecontours) > 0:
            blue_c = max(bluecontours, key=cv2.contourArea)
            size = cv2.contourArea(blue_c)
            if size > 1000:
                self.publish(self.image, 1, size, self.get_center(blue_c))

    def plum(self, maskimage):
        """
        Callback checking for Plum.

        Args:
            maskimage (numpy.ndarray): Masked image of the object
        
        Returns:
            None
        """

        purplecontours, _ = cv2.findContours(
            maskimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(purplecontours) > 0:
            purple_c = max(purplecontours, key=cv2.contourArea)
            size = cv2.contourArea(purple_c)
            if size > 1000:
                self.publish(self.image, 2, size, self.get_center(purple_c))

    def scarlet(self, maskimage):
        """Callback checking for Scarlet.

        Args:
            maskimage (numpy.ndarray): Masked image of the object
        
        Returns:
            None
        """

        redcontours, _ = cv2.findContours(
            maskimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(redcontours) > 0:
            red_c = max(redcontours, key=cv2.contourArea)
            size = cv2.contourArea(red_c)
            if size > 1000:
                self.publish(self.image, 3, size, self.get_center(red_c))

    def get_center(self, contour):
        """
        Gets the center of the contour.

        Args:
            contour (numpy.ndarray): The contour to get the center of

        Returns:
            tuple: The center of the contour
        """

        M = cv2.moments(contour)
        return int(M["m10"] / M["m00"])
