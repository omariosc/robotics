"""
You are asked to implement a program that controls a simulated Turtlebot to
find and identify a Cluedo character in an environment. In the environment,
there will be two rooms. Your robot needs to enter the “green room”, which
has a green circle near its entrance, and identify the character in the room.

The second room will be a “red room”, with a red circle near its entrance.
Your robot should not go into this room. You will work on this project as a
group. Your program will be tested using the Turtlebot Gazebo simulation in
several different worlds. Your program will be given a map of the environment
(you will learn about what a map is, how to build it and how to use it, in Lab
Session 5). Your robot will be placed at a start point, which will be the same
for all groups.

You will be given (x,y) coordinates of the entrance points of the two rooms in
the map. One room will have a red circle on the wall near its entrance, and
the other a green circle. The green/red circles on the walls will be visible from
these entrance points, but not necessarily from a direct angle.

Your robot will need to enter the room with the green circle on the door. You
will be given the (x,y) coordinates of the center points of both rooms. There
will be a Cluedo character in the green room and a Cluedo character in the red
room.

We know that the “murderer” is the Cluedo character in the green room, not
the one in the red room. We just need the robot to go into the green room and
tell us who s/he is.

Your robot, therefore, will have to find the Cluedo character in the green room
and report the identity of the character. In your group’s GitHub repo, we will
provide you with a set of images of different Cluedo characters and their names.
Your robot will need to identify which one is in the green room.

Character screenshot (1 point): When your robot thinks it saw the
image of the Cluedo character, it should save a snapshot of the camera
image with the filename “cluedo character.png”. 
The character must be completely contained within the saved image. 
If an image with this name is saved and it does not show the 
correct character from the green room, you will get -1 penalty point
(including if you take a screenshot of the wrong character in the
red room or take a screenshot of an empty wall).
Please make sure that you use “cluedo character.png” as the filename.

Character identification (1 point): Your program must then identify
the correct character in the green room, by printing out the character 
name into a text file with the filename “cluedo character.txt”.
If a file with this name is created, but includes a wrong character
name, you will get -1 penalty point. Please make sure that you use
“cluedo character.txt” as the filename.

Your robot will have at most 5 minutes per world to complete the task. If,
after 5 minutes of running, your program has not stopped by itself, it will be
stopped and the points you have collected up to that point in that run will be
your mark for that world.
"""

# Main file containing the logic for the robot to find and identify a Cluedo character in an environment.
from actionlib_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
from kobuki_msgs.msg import BumperEvent
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, Point, Pose, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import Int8, Int64, String
import cv2
import actionlib
import rospy
import yaml
import sys
import os
import numpy as np
import json
import math
import matplotlib
import time
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# ** Don't change this path **
PROJECT_DIR = os.path.expanduser('~/catkin_ws/src/group_project')

# Add the scripts directory to the path
sys.path.insert(0, f'{PROJECT_DIR}/scripts')

# **Import files in scripts from here ...**
from Timer import Timer
# from ColourIdentifier import ColourIdentifier
from CircleIdentifier import CircleIdentifier
from FrameIdentifier import FrameIdentifier


class Coursework():
    """The coursework class.

    The Coursework class is well structured and has already incorporated a number 
    of ROS functionalities such as topics, actionlib, and ROS parameters. 
    It's clear that it's designed to control a Turtlebot, 
    handle image processing tasks, and navigate in an environment.

    These are a set of functions that are responsible for different tasks, such as:
    - Reading configuration parameters from a YAML file
    - Initializing ROS subscribers and publishers
    - Reading input points of rooms from a YAML file
    - Manipulating a binary map to remove unnecessary rows and columns
    - Creating a map of the environment from OccupancyGrid data
    - Saving a screenshot of a detected character
    - Updating the size and the center of the detected character
    """

    def __init__(self):
        """
        Initialises the robot's ROS node and sets up necessary variables: speed, position, start time etc.
        
        Args:
            None
        """
        rospy.loginfo("Initialising node.")

        # Configuration parameters
        config = self.read_config()
        self.characters = config["characters"]
        # Dictionary of points
        self.points = {
            "r1c": None,
            "r2c": None,
            "r1e": None,
            "r2e": None
        }
        # Initialise the room entrance and centre points.
        self.read_input_points()
        # Store robot actual coordinate position
        self.robot_position = None
        # Dictionary of quaternions
        self.all_quaternions = {}
        # Compute all quaternions
        self.calculate_all_quaternions()
        # Circle Identifier Data
        self.circle_data = {
            "red": {"found": False, "centre": (0, 0), "size": (0, 0), "distance": math.inf},
            "green": {"found": False, "centre": (0, 0), "size": (0, 0), "distance": math.inf}
        }
        self.found = False
        self.map_data = None

        # Timer and ROS Rate
        self.rate = config["rate"]
        self.rate_limit = rospy.Rate(self.rate)
        self.timer = Timer(config["time_limit"])

        # Turtlebot parameters
        self.desired_velocity = Twist()
        self.linear_velocity = 0.2  # Set the robot's linear speed
        self.angular_velocity = 0.5236  # Set the robot's angular speed
        self.bumped = [None, None, None, None]  # Active?, left, middle, right

        # Subscribers and publishers
        self.sub_character_identified = None
        self.sub_character_area = None
        self.sub_character_center = None
        self.sub_text_identified = None
        self.sub_circle_identified = None
        self.sub_localisation = None
        self.sub_timer = None
        self.sub_bumper = None
        self.movement_publisher = None
        self.move_base = None
        self.goal_sent = False

        # Image parameters
        self.bridge = CvBridge()
        self.red_circle_identified = False
        self.green_circle_identified = False
        # Do not store image in case we are not in green room
        # and another image can be seen
        self.in_green_room = False
        self.image_shape = [640, 480]
        self.character_area = None
        self.character_center = None
        self.picture_saved = False
        self.pipeline_saved = False
        self.picture_centered = False
        self.picture_fit = False

        # What to do if shut down (e.g. Ctrl-C or failure)
        rospy.on_shutdown(self.shutdown)

    def read_config(self):
        """
        Reads the configuration file (config.yaml) containing parameters needed for the task.

        Args:
            None

        Returns:
            dict: The config data
        """

        path = f'{PROJECT_DIR}/config/config.yaml'
        rospy.loginfo(f"Reading config file from {path}...")
        with open(path, encoding="utf-8") as opened_file:
            return yaml.safe_load(opened_file)

    def init_subscribers(self):
        """
        Initialises the subscribers for topics such as identifying the image, circle, text; updating the robot's location and initialising its timer.

        Args:
            None

        Returns:
            None
        """
    
        self.sub_character_image_identified = rospy.Subscriber(
            'identified_character_image', Image, self.save_character_screenshot)
        self.sub_pipeline_image_identified = rospy.Subscriber(
            'identified_character_image_pipeline', Image, self.save_character_pipeline_screenshot)
        self.sub_text_identified = rospy.Subscriber(
            'identified_name', Int8, self.write_character_id)
        self.sub_character_area = rospy.Subscriber(
            'identified_area', Int64, self.update_character_area)
        self.sub_character_center = rospy.Subscriber(
            'identified_center', Int64, self.update_character_center)
        self.sub_circle_identified = rospy.Subscriber(
            'identified_circle', String, self.update_circle_identified)
        self.sub_localisation = rospy.Subscriber(
            "/amcl_pose", PoseWithCovarianceStamped, self.update_robot_location)
        self.sub_bumper = rospy.Subscriber(
            '/mobile_base/events/bumper', BumperEvent, self.bumper_handler)
        self.sub_timer = rospy.Subscriber('timer', Int8, self.break_condition)

        # Subscribe to the map topic.
        rospy.Subscriber("/map", OccupancyGrid, self.create_map)

        rospy.loginfo("Subscribers initialised.")

    def init_publishers(self):
        """
        Initialises the publishers for the robot's velocity and move_base used to send commands to the robot to move to a specific location.

        Args:
            None

        Returns:
            None
        """
        self.movement_publisher = rospy.Publisher(
            'mobile_base/commands/velocity', Twist, queue_size=0)

        # Create a new action client
        self.move_base = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)
        
        self.pub_shutdown = rospy.Publisher(
            'execution_shutdown', Int8, queue_size=0)

        # Wait for the action server to become available
        self.move_base.wait_for_server()

        rospy.loginfo("Publishers initialised.")

    def read_input_points(self, debug=False):
        """
        Reads the co-ordinates of the entrance points of the rooms and their center points.

        Args:
            debug (bool, optional): Whether to print the input points. Defaults to False.

        Returns:
            None
        """

        # Set up path and read the co-ordinates from the yaml file
        path = f'{PROJECT_DIR}/world/input_points.yaml'
        points = None
        try:
            with open(path, encoding="utf-8") as opened_file:
                points = yaml.safe_load(opened_file)
        except NotADirectoryError:  # If running main.sh and world dir was deleted
            path = f'{PROJECT_DIR}/world'
            rospy.loginfo(f"Reading input points from {path}...")
            with open(path, encoding="utf-8") as opened_file:
                points = yaml.safe_load(opened_file)
        finally:
            if debug:
                rospy.loginfo(f"Input points: {points}.")

        # Assign the co-ordinates
        self.points = {
            "r1c": Point(
                points["room1_centre_xy"][0], points["room1_centre_xy"][1], 0),
            "r2c": Point(
                points["room2_centre_xy"][0], points["room2_centre_xy"][1], 0),
            "r1e": Point(
                points["room1_entrance_xy"][0], points["room1_entrance_xy"][1], 0),
            "r2e": Point(
                points["room2_entrance_xy"][0], points["room2_entrance_xy"][1], 0)
        }

    def show_map(self, binary_map):
        """
        Displays the cropped binary map by removing rows and columns containing only zeros.
        
        Args:
            binary_map (numpy.ndarray): The binary map to be displayed.
            
        Returns:
            numpy.ndarray: The cropped binary map.
        """

        """
        In this updated function, np.all checks if all the elements in the specified row or column are equal to 0. 
        If they are, that row or column is removed from binary_map. 
        This continues until a row or column with at least a 1 or 0.5 is encountered. 
        The slicing operations remove the appropriate rows or columns.

        This function removes rows and columns from all four sides of the map, 
        effectively cropping out large areas of zeros. 
        However, it leaves a border of zeros around the map as requested.
        """

        # Remove top rows
        while np.all(binary_map[0] == 0):
            binary_map = binary_map[1:]

        # Remove bottom rows
        while np.all(binary_map[-1] == 0):
            binary_map = binary_map[:-1]

        # Remove left columns
        while np.all(binary_map[:, 0] == 0):
            binary_map = binary_map[:, 1:]

        # Remove right columns
        while np.all(binary_map[:, -1] == 0):
            binary_map = binary_map[:, :-1]

        # plt.imshow(binary_map, cmap='gray')
        # plt.show()

        return binary_map

    def create_map(self, data):
        """
        Creates a map of the environment.
        
        Args:
            data (numpy.ndarray): The input data representing the map
            
        Returns:
            None
        """

        # Check map data does not already exist
        if self.map_data is not None:
            return

        # Convert map data to 2D array
        self.map_data = data
        map_2d = np.reshape(
            self.map_data.data, (self.map_data.info.height, self.map_data.info.width))

        # Create a binary map from the 2D array
        binary_map = np.zeros(
            (self.map_data.info.height, self.map_data.info.width))
        for i in range(self.map_data.info.height):
            for j in range(self.map_data.info.width):
                if map_2d[i][j] == 100:
                    binary_map[i][j] = 1
                elif map_2d[i][j] == -1:
                    binary_map[i][j] = 0
                else:
                    binary_map[i][j] = 0.5

        # Draw the binary map
        # plt.imshow(binary_map, cmap='gray')
        # plt.show()

        self.show_map(binary_map)

    def write_character_id(self, image_id):
        """
        Writes the identified Character's ID to a file.

        Args:
            image_id (_type_): The character id
            
        Returns:
            None
        """

        if not self.in_green_room:
            return

        path = f'{PROJECT_DIR}/output/cluedo_character.txt'
        # Write character ID
        with open(path, 'w', encoding="utf-8") as opened_file:
            opened_file.write(self.characters[image_id.data])
            rospy.loginfo(
                f'Found "{self.characters[image_id.data]}" at {self.character_center}.')
            
            self.found = True
            
            # In case the node is shutdown before the image file is written
            # Call shutdown manually
            # rospy.signal_shutdown("Found character.")

    def save_character_pipeline_screenshot(self, ros_image):
        """
        Saves a screenshot of the pipeline stages when identifying a character:
            1. Original Image
            2. Grayscale
            3. Gaussian Blur
            4. Thresholding
            5. Canny Edge Detector
            6. Overlayed Result Image

        Args:
            ros_image (sensor_msgs.msg.Image): The screenshot to save

        Returns:
            None
        """

        # if self.pipeline_saved:
        #     return
        
        try:
            self.save_screenshot(ros_image, f'{PROJECT_DIR}/output/cluedo_character_pipeline.png')
            self.pipeline_saved = True
        except CvBridgeError as e:
            print(e)
        
    def save_character_screenshot(self, ros_image):
        """
        Saves the raw camera image from the turtlebot (which should have the character).

        Args:
            ros_image (sensor_msgs.msg.Image): The screenshot to save.

        Returns:
            None
        """
        
        # if self.picture_saved:
        #     return

        try:
            self.save_screenshot(ros_image, f'{PROJECT_DIR}/output/cluedo_character.png')
            self.picture_saved = True
        except CvBridgeError as e:
            print(e)

    def save_screenshot(self, ros_image, path=None):
        """
        Saves a screenshot.

        Args:
            ros_image (sensor_msgs.msg.Image): The screenshot to save.

        Returns:
            None
        """
        
        if not self.in_green_room: # and not self.found:
            return

        # Cancel any active goals.
        if self.goal_sent:
            self.move_base.cancel_goal()

        # if not self.picture_fit or not self.picture_centered:
        #     return

        # Convert the received image into a opencv image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            cv2.imwrite(os.path.expanduser(path), cv_image)
            rospy.loginfo(f"Saved screenshot to {path}.")
        except CvBridgeError as error:
            print(error)

    def update_character_area(self, area):
        """Updates the character size.

        Args:
            area (_type_): The area of the character image
        
        Returns:
            None
        """

        if not self.in_green_room:
            return

        self.character_area = area.data

    def update_character_center(self, center):
        """Update the character center.

        Args:
            center (_type_): The character center

        Returns:
            None
        """

        if not self.in_green_room or self.character_area is None:
            return

        self.character_center = center.data
        self.found = True

    def simple_output(self, i):
        """Callback function for testing. Outputs any input.

        Args:
            i (string): The input to output

        Returns:
            None
        """

        rospy.loginfo(i)

    def update_circle_identified(self, circle_data):
        """Updates the circle identified.

        Args:
            circle_data (dict): The circle identified
        
        Returns:
            None
        """

        """
        These functions are callback methods for the ROS subscribers. 
        They update the status of the robot based on the data received from the relevant topics.

        This method is a ROS subscriber callback that updates the circle_data attribute of the class. 
        This attribute is a Python dictionary that's derived from a 
        JSON string received from the 'identified_circle' topic.
        """

        # Convert the json String to a python dictionary
        self.circle_data = json.loads(circle_data.data)

    def update_robot_location(self, position):
        """Updates the robot's location.

        Args:
            position (Point): The robot location

        Returns:
            None
        """

        """
        These functions are callback methods for the ROS subscribers. 
        They update the status of the robot based on the data received from the relevant topics.

        This is another ROS subscriber callback that updates the robot_position 
        attribute using the position data received from the "/amcl_pose" topic.
        """

        self.robot_position = Point(
            position.pose.pose.position.x, position.pose.pose.position.y, position.pose.pose.position.z)

    def bumper_handler(self, data):
        """Handles the bumper input.

        Args:
            data (BumperEvent): The bumper data
        
        Returns:
            None
        """

        """
        This method handles bumper events, updating the bumped 
        attribute  based on the bumper event data received 
        from the '/mobile_base/events/bumper' topic.
        """

        if data.state == BumperEvent.PRESSED:
            self.bumped[0] = True
        elif data.state == BumperEvent.RELEASED:
            self.bumped[0] = False

        if data.bumper == BumperEvent.LEFT:
            self.bumped[1] = True
        if data.bumper == BumperEvent.CENTER:
            self.bumped[2] = True
        if data.bumper == BumperEvent.RIGHT:
            self.bumped[3] = True

    def stop_robot(self):
        """Stops the robot.

        Args:
            None
            
        Returns:
            None
        """

        """
        This method publishes zero velocity commands to 
        the 'mobile_base/commands/velocity' topic, 
        effectively stopping the robot.
        """

        for _ in range(10):
            self.desired_velocity.linear.x = 0
            self.desired_velocity.angular.z = 0
            self.movement_publisher.publish(self.desired_velocity)
            self.rate_limit.sleep()

    def break_condition(self, timer):
        """Checks if the break condition has been met.

        Args:
            timer (Timer): The timer

        Returns:
            bool: Whether the break condition has been met
        """

        """
        This function checks if a break condition, such as a time limit, has been met, and if so, it initiates a shutdown.

        This is a callback method that checks if a break condition (like a time limit) has been met. 
        If the condition has been met, the method initiates a shutdown.
        """

        # Check if time limit has been reached
        if timer.data == 1:
            rospy.signal_shutdown("Did not find character in time.")

    def pixel_to_world(self, pixel_coordinates):
        """
        Converts pixel coordinates to world coordinates.
        
        Args:
            pixel_coordinates (tuple): The pixel coordinates

        Returns:
            tuple: The world coordinates
        """

        x_pixel, y_pixel = pixel_coordinates

        x_world = self.map_data.info.origin.position.x + \
            (x_pixel * self.map_data.info.resolution)
        y_world = self.map_data.info.origin.position.y + \
            (y_pixel * self.map_data.info.resolution)

        return (x_world, y_world)

    def world_to_pixel(self, world_coordinates):
        """
        Converts world coordinates to pixel coordinates.
        
        Args:
            world_coordinates (tuple): The world coordinates

        Returns:
            tuple: The pixel coordinates
        """

        x_world, y_world = world_coordinates

        x_pixel = int(
            (x_world - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        y_pixel = int(
            (y_world - self.map_data.info.origin.position.y) / self.map_data.info.resolution)

        return (x_pixel, y_pixel)

    def move_to_position(self, position, quaternion, debug=True):
        """Moves the robot to a position. 

        Args:
            position (Point): The position to move to
            quaternion (Quaternion): The quaternion to direct the robot
            debug (bool): Whether to output debug information
        
        Returns:
            None
        """

        """
        This function is taken from the GoToPose class in the ROS tutorials. 
        It is used to move the robot to a specific position.
        """

        rospy.loginfo(f'Moving to {position.x}, {position.y}...')

        # For robustness we store if a goal was sent.
        # If a goal was sent and the ros shutdowns
        # we can cancel the goal.
        self.goal_sent = True

        # Create a goal state
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        # Set the goal state to the given position and quaternion
        goal.target_pose.pose = Pose(
            Point(position.x, position.y, position.z),
            Quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        )

        # Send the goal state to the action server
        self.move_base.send_goal(goal)

        # Allow TurtleBot up to 60 seconds to complete task
        success = self.move_base.wait_for_result(rospy.Duration(60))

        state = self.move_base.get_state()
        result = False

        if success and state == GoalStatus.SUCCEEDED:
            result = True
        else:
            self.move_base.cancel_goal()
            rospy.loginfo('Cancelled goal.')

        self.goal_sent = False

        # Only log result if debugging
        if result and debug:
            rospy.loginfo('Reached the desired pose.')
        else:
            rospy.loginfo('Failed to reach the desired pose.')

    def move_to_pixel_position(self, pixel_coordinates, quaternion, debug=True):
        """Moves the robot to a position specified by pixel coordinates.
        
        Args:
            pixel_coordinates (tuple): The pixel coordinates
            quaternion (Quaternion): The quaternion to direct the robot
            debug (bool): Whether to output debug information
        
        Returns:
            None
        """

        # Convert pixel coordinates to world coordinates
        world_coordinates = self.pixel_to_world(pixel_coordinates)
        x_world, y_world = world_coordinates

        # Create a position message
        position = Point()
        position.x = x_world
        position.y = y_world
        position.z = 0.0

        # Call the existing function to move to the specified position
        self.move_to_position(position, quaternion, debug)

    def calculate_quaternions(self, position1, position2):
        """Calculates the quaternions to represent a rotation from one point to another.

        Args:
            position1 (Point): The first position
            position2 (Point): The second position
        
        Returns:
            Quaternion: The quaternion representing the rotation
        """

        """
        Given two points in space, this function calculates the quaternion 
        (a representation of rotation in 3D space) that represents the 
        rotation from the first point to the second.
        """
        # calculate the vector between the two points
        vector = Point(position2.x - position1.x, position2.y -
                       position1.y, position2.z - position1.z)
        # calculate the angle to face the second point
        angle = math.atan2(vector.y, vector.x)

        # calculate the quaternions
        quaternion = Quaternion(0, 0, math.sin(angle/2), math.cos(angle/2))

        return quaternion

    def calculate_all_quaternions(self):
        """Calculates and stores the quaternions for all possible pairs of points.

        Args:
            None

        Returns:
            None
        """

        """
        To compute the quaternions between all possible pairs of points in O(1) time, 
        you can precompute these quaternions and store them in a Python dictionary. 
        The keys of this dictionary can be tuples that represent pairs of points, 
        and the values can be the corresponding quaternions. 

        You can call calculate_all_quaternions after you have initialized all your points. 
        Then, you can get the quaternion for any pair of points by calling 
        get_quaternion with the names of those points as arguments.

        Note: The calculate_all_quaternions function assumes that the points are 
        attributes of the class instance and their names are given as strings. 
        If the points are stored in a different way, 
        you would need to adjust this function accordingly.
        """

        for point1_name, point1 in self.points.items():
            for point2_name, point2 in self.points.items():
                if point1_name != point2_name:
                    # Calculate the quaternion and store it in the dictionary
                    self.all_quaternions[(point1_name, point2_name)] = self.calculate_quaternions(
                        point1, point2)

    def get_quaternion(self, point1_name, point2_name):
        """Gets the precomputed quaternion for a pair of points in O(1) time.
        
        Args:
            point1_name (str): The name of the first point
            point2_name (str): The name of the second point
        
        Returns:
            Quaternion: The quaternion representing the rotation
        """

        return self.all_quaternions[(point1_name, point2_name)]

    def rotate_to_angle(self, angle=2*math.pi, initial=False, circle=True):
        """Rotates the robot to a given angle.

        Args:
            angle (float): The angle to rotate to
            initial (bool, optional): Whether this is the initial rotation. Defaults to False.
            circle (bool, optional): Whether to search for a circle. Defaults to True.

        Returns:
            string: The colour of the circle found
            None: If no circle was found
        """

        """
        This function allows the robot to rotate to a specified angle in order to search for a circle.
        """

        if circle:
            rospy.loginfo("Searching for circle...")

        self.desired_velocity.linear.x = 0.0
        # Turn with 0.5236 rad/sec. (12 seconds for 2pi)
        self.desired_velocity.angular.z = self.angular_velocity
        # Rotate to angle
        rospy.loginfo(f"Rotating to {angle} radians...")
        for _ in range(int(angle / self.angular_velocity) * 10):
            self.movement_publisher.publish(self.desired_velocity)
            self.rate_limit.sleep()
            # If green circle found, return "green"
            if self.circle_data["green"]["found"] and circle:
                rospy.loginfo("Green circle found.")
                return "green"
            # If red circle found, return "red"
            if self.circle_data["red"]["found"] and not initial and circle:
                rospy.loginfo("Red circle found.")
                return "red"
            
            if self.found:
                return None

        # Stop turning
        rospy.loginfo("Stopping rotation...")
        self.stop_robot()

        if circle:
            # if no circle found, return None
            rospy.loginfo("No circle found.")
            return None

    def search_for_character(self, angle=2*math.pi):
        """Searches for a character.

        In order to search for a character in the green room, the robot needs to explore the room in a methodical way. 
        The robot should also mark the area around it as seen in the map. 
        A common strategy for this kind of problem is using a breadth-first search (BFS). 
        Here's a simple approach to implement this:

        1. Create a new 2D array (same size as the map) to store whether a cell has been seen or not.
        2. Implement a BFS from the robot's current position. For each position the robot visits, mark the surrounding 5x5 square as seen.
        3. When choosing the next position to visit, pick from the unvisited neighbors of all currently visited positions, giving priority to those closest to the robot's current position. If there are no unvisited neighbors, move to the closest unvisited cell.
       
        Args:
            angle (float): The angle to rotate to

        Returns:
            None

        """

        rospy.loginfo("Searching for character...")
        self.in_green_room = True

        self.desired_velocity.linear.x = 0
        # Turn with 0.5236 rad/sec. (12 seconds for 2pi)
        self.desired_velocity.angular.z = self.angular_velocity
        for _ in range(int(angle / self.angular_velocity) * 10):
            self.movement_publisher.publish(self.desired_velocity)
            self.rate_limit.sleep()

            if self.found:
                return

        # Stop turning
        rospy.loginfo("Stopping rotation...")
        self.stop_robot()

        # Rotate 135 degrees to the left
        self.rotate_to_angle(3*math.pi/4, initial=True, circle=False)

        t1 = time.time()

        while not self.found:
            # Move foward until bumper is pressed
            self.desired_velocity.linear.x = 0.2
            self.desired_velocity.angular.z = 0.0
            while not self.found and not self.bumped[0]:
                self.movement_publisher.publish(self.desired_velocity)
                self.rate_limit.sleep()
            self.desired_velocity.linear.x = 0.0
            self.desired_velocity.angular.z = 0.2
            while not self.found and self.bumped[0]:
                self.movement_publisher.publish(self.desired_velocity)
                self.rate_limit.sleep()

            if int(time.time() - t1) % 10 == 0:
                self.rotate_to_angle(2*math.pi, initial=True, circle=False)



            

        # # Rotate so that the robot is facing the character and move forward so character size is not too large
        # rospy.loginfo("Aligning robot to image...")
        # while not self.picture_centered or not self.picture_fit:
        #     # log size and center
        #     # rospy.loginfo(
        #     #     f"Size: {self.character_area}, Center: {self.character_center}")
        #     if self.character_center > self.image_shape[0] + 30:
        #         self.picture_centered = False
        #         self.desired_velocity.angular.z = -0.2
        #     elif self.character_center < self.image_shape[0] - 30:
        #         self.picture_centered = False
        #         self.desired_velocity.angular.z = 0.2
        #     else:
        #         self.picture_centered = True
        #         self.desired_velocity.angular.z = 0.0

        #     self.picture_centered = True
        #     self.desired_velocity.angular.z = 0.0

        #     if self.character_area > 55:
        #         self.picture_fit = False
        #         self.desired_velocity.linear.x = -0.1
        #     elif self.character_area < 35:
        #         self.picture_fit = False
        #         self.desired_velocity.linear.x = 0.1
        #     else:
        #         self.picture_fit = True
        #         self.desired_velocity.linear.x = 0.0

        #     self.movement_publisher.publish(self.desired_velocity)
        #     self.rate_limit.sleep()

        # Stop moving
        rospy.loginfo("Stopping movement...")
        self.stop_robot()

        self.in_green_room = False

        rospy.loginfo("Stopped search for character.")

        # # Initialize the seen array
        # seen = np.zeros((height, width))

        # # Use a queue for the BFS, starting at the robot's current position
        # queue = [(self.robot_position.x, self.robot_position.y)]

        # while queue:
        #     # Get the next position from the queue
        #     x, y = queue.pop(0)

        #     # Mark the surrounding 5x5 square as seen
        #     seen[max(0, x - 2):min(height, x + 3), max(0, y - 2):min(width, y + 3)] = 1

        #     # Iterate over the position's neighbors
        #     for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        #         nx, ny = x + dx, y + dy

        #         # If the neighbor is inside the map and not seen yet, add it to the queue
        #         if 0 <= nx < height and 0 <= ny < width and not seen[nx][ny]:
        #             queue.append((nx, ny))

        #     # Move the robot to the next position
        #     # Implement movement control here

    def move_room_1(self):
        """Moves robot to room 1.
        
        Args:
            None

        Returns:
            None
        """

        rospy.loginfo("Moving to room 1...")

        # Multiple if conditionals to check for early termination
        if not self.found:
            # Move to room 1
            self.move_to_position(
                self.points["r1c"], self.all_quaternions[("r1e", "r1c")])
            
        if not self.found:
            # Search for character
            self.search_for_character()

    def move_room_2(self):
        """Moves robot to room 2.
        
        Args:
            None

        Returns:
            None
        """

        rospy.loginfo("Moving to room 2...")
        # Move to room 2
        self.move_to_position(
            self.points["r2c"], self.all_quaternions[("r2e", "r2c")])
        # Search for character
        self.search_for_character()

    def contingency_plan(self):
        """
        Invoked when robot is unable to find a character. Moves robot to a different location and searches again.

        Args:
            None

        Returns:
            None
        """

        rospy.loginfo("Executing contingency plan...")
        self.move_room_2()
        # If not found then leave and go to room 1 center
        self.move_room_1()

    def move_to_circle(self):
        """
        Moves robot to the circle in the green room.

        Args:
            None

        Returns:
            None
        """
        rospy.loginfo("Moving to green circle...")

        # Get circle center and size
        (x,
         _), size = self.circle_data["green"]["centre"], self.circle_data["green"]["size"]

        # Use a wider range for x before deciding to rotate
        range_threshold = self.image_shape[0] // 4

        while size >= 255:
            # If the circle is to the left of the center, rotate right
            if x < self.image_shape[0] // 2 - range_threshold:
                self.desired_velocity.linear.x = 0.1
                self.desired_velocity.angular.z = 0.2
            # If the circle is to the right of the center, rotate left
            elif x > self.image_shape[0] // 2 + range_threshold:
                self.desired_velocity.linear.x = 0.1
                self.desired_velocity.angular.z = -0.2
            # If the circle is at the center, don't rotate
            else:
                self.desired_velocity.linear.x = 0.2
                self.desired_velocity.angular.z = 0

            # Publish the desired velocity
            self.movement_publisher.publish(self.desired_velocity)
            self.rate_limit.sleep()

        # Stop moving
        self.stop_robot()

        # Now use the exact robot position to determine whether room 1 or room 2 entrance is closer
        if self.robot_position - self.points["r1e"] < self.robot_position - self.points["r2e"]:
            self.move_room_1()
        else:
            self.move_room_2()

    def main_logic(self):
        """
        Guides the robot's actions based on whether or not it finds certain "circles" and their colors in different rooms.

        Args:
            None

        Returns:
            None
        """

        # Begin by searching for green circle
        # circle_found_initial = self.rotate_to_angle(initial=True)
        # if circle_found_initial == "green":
        #     # Move to circle and search for character
        #     self.move_to_circle()
        #     # If still running, call shutdown manually
        #     rospy.signal_shutdown("Did not find character in the green room.")
        #     return

        # Move to room 1 entrance
        rospy.loginfo("Moving to room 1 entrance...")
        self.move_to_position(self.points["r1e"], self.all_quaternions[(
            "r1e", "r1c")])
        # Look for circle
        circle_found_room1 = self.rotate_to_angle()

        # if no circle or red circle, move to room 2 entrance
        if not circle_found_room1:
            self.move_to_position(self.points["r2e"], self.all_quaternions[(
                "r2e", "r2c")])

            # Look for circle
            circle_found_room2 = self.rotate_to_angle()

            # If red circle found then go back to room 1
            if circle_found_room2 == "red":
                self.move_room_1()
            # If green circle found then go to room 2
            elif circle_found_room2 == "green":
                self.move_room_2()
            # If no circle found then execute contingency plan
            else:
                self.contingency_plan()

        # If red circle move to room 2 center
        elif circle_found_room1 == "red":
            self.move_room_2()

        # If green circle move to room 1 center
        elif circle_found_room1 == "green":
            self.move_room_1()

        self.pub_shutdown.publish(42)

        # If still running, call shutdown manually
        rospy.signal_shutdown("Did not find character in the green room.")

    def run(self):
        """
        Runs the main loop; sets up subscribers, publishers, initialises vars and runs the main logic.

        Args:
            None

        Returns:
            None   
        """

        # Initialise the subscribers and publishers
        self.init_subscribers()
        self.init_publishers()

        # Run the main logic.
        self.main_logic()

        # Cancel the timer if still running
        self.timer.shutdown()

        # If still running, call shutdown manually
        rospy.signal_shutdown("Did not find character in time.")

    def shutdown(self, reason=""):
        """
        Shuts down the robot, cancels any active goals, and closes all windows.

        Args:
            reason (str): The reason for shutting down.

        Returns:
            None
        """
        # Cancel any active goals.
        if self.goal_sent:
            self.move_base.cancel_goal()

        # Stop the robot.
        self.desired_velocity.linear.x = 0
        self.desired_velocity.angular.z = 0
        for _ in range(10):
            self.movement_publisher.publish(self.desired_velocity)
            self.rate_limit.sleep()

        # Close all windows.
        cv2.destroyAllWindows()
        rospy.loginfo(reason)
        rospy.sleep(1)
        sys.exit(0)


if __name__ == '__main__':
    """
    The if __name__ == '__main__' block at the end of the script 
    is the entry point when the script is run directly. 
    It initializes the ROS node, creates instances of the 
    Coursework, ColourIdentifier, and CircleIdentifier classes, 
    and calls the run() method of the Coursework instance.
    """

    try:
        # Initialise main node.
        rospy.init_node('group_project')

        # Initialise the main coursework class (this is a higher level class that will utilise
        # others to get a final result.
        coursework = Coursework()

        # The debug flag controls whether images etc. are shown 
        # using opencv. 
        # colour_identifier = ColourIdentifier() 
        circle_identifier = CircleIdentifier(debug=True) 
        frame_identifier = FrameIdentifier(debug=True, project_dir=PROJECT_DIR) 


        # Run the main loop.
        coursework.run()

    except KeyboardInterrupt:
        # This block of code will run when a keyboard interrupt (CTRL+C) is detected.

        # Close any open windows.
        cv2.destroyAllWindows()
        print("Interrupt received, closing...")
        sys.exit(1)
