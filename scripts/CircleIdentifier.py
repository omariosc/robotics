# Contains the class used to identify the shape of an object in an image to detect red/green circles.

import math

import rospy
import cv2
from math import inf
import numpy as np
import json

from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError


class CircleIdentifier:
    
    """
    Creates a circle identifier object used to detect red and green circles

    Args:
        debug (bool, optional): Whether debug code will be executed. Defaults to False.
    
    Returns:
        CircleIdentifier: The circle identifier object.
    """
    def __init__(self, debug=False):
        # Sensitivity of the hsv mask ranges.
        self.sensitivity = 20

        # Controls the debug image shown in the window.
        self.debug_image = np.zeros((100, 100, 3))

        # Lower and upper bounds for red and green circles.
        # The hue values are in the range of 0-127 (opencv)
        # - Red circle HSV colour: hsl(0, 100%, 32%)
        # - Green circle HSV colour: hsl(73.5, 100%, 22%)
        self.hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        self.hsv_green_lower = np.array([60 - self.sensitivity, 100, 40])
        self.hsv_red_upper = np.array([0 + self.sensitivity, 255, 255])
        self.hsv_red_lower = np.array([0, 100, 60])

        # Will control whether debug code (logging, showing images etc.) will be executed.
        self.debug = debug

        # Image from the camera.
        self.image = None

        # Flag to control has the circle been found.
        self.circle_found = False

        # Initialize circle data
        self.circle_data = {
            "red": {"found": False, "centre": (0, 0), "size": 0, "distance": inf},
            "green": {"found": False, "centre": (0, 0), "size": 0, "distance": inf}
        }

        # Initialize the distance to the circle.
        self.distance_to_circle = -1

        # Subscribe to the camera output from the turtlebot.
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/rgb/image_raw", Image,
                         self.callback_process_image)

        # Subscribe to the laser scan of the turtlebot.
        rospy.Subscriber("/scan", LaserScan, self.callback_process_laser_scan)

        # Publisher to broadcast a message that a circle has been identified.
        self.identified_circle_publisher = rospy.Publisher(
            'identified_circle', String, queue_size=0)

    def callback_process_laser_scan(self, data):
        """
        Processes the laser scan data and calculates the distance to the circle.
        Args:
            data (sensor_msgs.msg.LaserScan): Scanned data from the laser scan.
        """
        self.distance_to_circle = data.ranges[len(data.ranges)//2]

    def callback_process_image(self, data):
        """
        Processes the received image to see if a circle has been detected.

        Args:
            data (sensor_msgs.msg.Image): The received image
        """

        # Convert the received image into a opencv image
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as error:
            print(error)

        # Convert the rgb image into a hsv image
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Creating the masks for red and green circles.
        green_mask = cv2.inRange(
            hsv_image, self.hsv_green_lower, self.hsv_green_upper)
        red_mask = cv2.inRange(
            hsv_image, self.hsv_red_lower, self.hsv_red_upper)

        combined_mask_red_green = cv2.bitwise_or(red_mask, green_mask)
        thresholded_image = cv2.bitwise_and(
            self.image, self.image, mask=combined_mask_red_green)

        # See if there is a circle in the view.
        self.detect_circle_using_contours(thresholded_image)

        if self.circle_found:
            circle_data_json = json.dumps(self.circle_data)
            self.identified_circle_publisher.publish(circle_data_json)

        # Show the debug image if we are in debug mode.
        # if self.debug:
        #     cv2.namedWindow('[CircleIdentifier] Debug Output')
        #     cv2.imshow('[CircleIdentifier] Debug Output', self.debug_image)
        #     cv2.waitKey(3)

    def detect_circle_using_contours(self, thresholded_image=None, min_shape_factor=0.82, max_shape_factor=1.2):
        """
        Detects a circle and returns information about the detected circle(s).

        Args:
            thresholded_image (numpy.ndarray, optional): The thresholded image used for circle detection.
            min_shape_factor (float, optional): The minimum shape factor to consider when fitting an ellipse.
            max_shape_factor (float, optional): The maximum shape factor to consider when fitting an ellipse.

        Returns:
            dict: A dictionary containing information about the detected circle(s).
        """

        # Convert the image to a grayscale image, as cv2.HoughCircles only accepts grayscale
        # (i.e.) single channel image. The reason we do not convert this from the process
        # callback is that we need to use the colour of the circle once they are detected.
        thresholded_image_grayscale = cv2.cvtColor(
            thresholded_image, cv2.COLOR_BGR2GRAY)

        # List of ellipses to store any ellipses found.
        ellipse_list = []

        # Reset the circle data.
        self.circle_data = {
            "red": {"found": False, "centre": (0, 0), "size": 0, "distance": 0},
            "green": {"found": False, "centre": (0, 0), "size": 0, "distance": 0}
        }

        # Look for circles using the cv2.findContours() function. We are using cv2.RETR_EXTERNAL
        # because we just want the outside contours of the object.
        potential_contours, _ = cv2.findContours(thresholded_image_grayscale, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
        if len(potential_contours) > 0:
            for potential_contour in potential_contours:
                # We are expecting the circle to be a closed shape.
                contour_perimeter = cv2.arcLength(
                    potential_contour, closed=True)

                contour_area = cv2.contourArea(potential_contour)

                # Initialise shape_factor to an invalid state.
                shape_factor = -1
                divisor = math.pow(contour_perimeter, 2.0)

                # Prevent divide by zero error.
                if divisor > 0.0:
                    # Shape factor formula from: http://www.empix.com/NE%20HELP/functions/glossary/morphometric_param.htm
                    shape_factor = (4.0 * math.pi * contour_area) / divisor

                # Fit an ellipse, if we think the shape is an ellipsoid.
                if min_shape_factor <= shape_factor <= max_shape_factor:
                    ellipse_list.append(cv2.fitEllipse(potential_contour))

        # We are making a copy here because we don't want to accidentally
        # alter the original image used for calculations.
        ellipse_visulisation_image = np.copy(thresholded_image)
        circle_mask = np.zeros_like(thresholded_image, dtype='uint8')
        # Reset the circle found flag.
        self.circle_found = False

        # Make sure we found at least one circle.
        if len(ellipse_list) > 0:
            for ellipse in ellipse_list:
                # Visualise the circle if we are in debug mode.
                cv2.ellipse(ellipse_visulisation_image,
                            ellipse, (255, 255, 255), 5)

                # Make a mask to only retain pixels inside the circles.
                cv2.ellipse(circle_mask, ellipse, (255, 255, 255), -1)

                # Convert circle mask to grayscale
                if len(circle_mask.shape) == 3:  # if the image has more than one channel
                    circle_mask = cv2.cvtColor(circle_mask, cv2.COLOR_BGR2GRAY)

                # Determine colour of circle.
                # Work out the mean hue inside the circle.
                mean_hue_inside_detected_circles = cv2.mean(
                    thresholded_image, mask=circle_mask)[0]

                if 1 <= mean_hue_inside_detected_circles <= 10:
                    self.circle_data["red"]["found"] = True
                    self.circle_data["red"]["centre"] = ellipse[0]
                    self.circle_data["red"]["size"] = (
                        ellipse[0][0] - 320) ** 2 + (ellipse[0][1] - 240) ** 2
                    self.circle_found = True
                    rospy.loginfo("Found green circle")
                    # If centre is within threshold of 15 pixels of center of image, then store distance
                    if self.circle_data["red"]["size"] <= 225:
                        rospy.loginfo("Distance to red circle: %s",
                                      self.distance_to_circle)
                        self.circle_data["red"]["distance"] = self.distance_to_circle
                elif 45 <= mean_hue_inside_detected_circles <= 55:
                    self.circle_data["green"]["found"] = True
                    self.circle_data["green"]["centre"] = ellipse[0]
                    self.circle_data["green"]["size"] = (
                        ellipse[0][0] - 320) ** 2 + (ellipse[0][1] - 240) ** 2
                    self.circle_found = True
                    if self.circle_data["green"]["size"] <= 225:
                        rospy.loginfo("Distance to green circle: %s",
                                      self.distance_to_circle)
                        self.circle_data["green"]["distance"] = self.distance_to_circle

        self.debug_image = ellipse_visulisation_image

        return self.circle_data

    # TODO: Did not give very good results - not very robust.
    def detect_circle_using_hough_circles(self, thresholded_image=None):
        """
        Detects a circle. Returns true if circle is detected else false.

        Args:
            thresholded_image (numpy.ndarray, optional): The thresholded image to use for circle detection.
        """

        # Convert the image to a grayscale image, as cv2.HoughCircles only accepts grayscale
        # (i.e.) single channel image.

        thresholded_image_grayscale = cv2.cvtColor(
            thresholded_image, cv2.COLOR_BGR2GRAY)

        # Look for circles using OpenCV.
        # dp: Controls the size of the accumulator array. Lower values -> Higher resolution.
        # minDist: The minimum distance between two circles. The idea is to keep this large,
        # so that we only detect one circle at a time.
        hough_circles_result = cv2.HoughCircles(
            thresholded_image_grayscale, cv2.HOUGH_GRADIENT, 1.2, 100)

        # Visualise the circle if we are in debug mode.
        if self.debug:
            # Make sure we found at least one circle.
            if hough_circles_result is not None:
                print('circle found')
                # We are making a copy here because we don't want to accidentally
                # alter the original image used for calculations.
                circle_visulisation_image = np.copy(thresholded_image)

                for hough_circle in hough_circles_result:
                    cv2.circle(img=circle_visulisation_image, center=(hough_circle[0], hough_circle[1]),
                               radius=hough_circle[2], color=(255, 255, 255), thickness=5)

                self.debug_image = circle_visulisation_image

        return False

    # Template matching?
