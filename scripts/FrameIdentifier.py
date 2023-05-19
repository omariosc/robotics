import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Int8, Int64
from cv_bridge import CvBridge, CvBridgeError


class FrameIdentifier:
    """Creates a frame identifier object.
    This class will be used for for detecting and identifying 
    the four different Cluedo characters.
    """

    def __init__(self, debug=False, project_dir=None):
        # Sensitivity of the hsv mask ranges.
        self.sensitivity = 35
        self.project_dir = project_dir

        # Controls the debug images shown in the window.
        # cv2.destroyAllWindows()
        self.cv_pipeline_image = np.zeros((100, 100, 3))
        self.debug_image_1 = np.zeros((100, 100, 3))
        self.debug_image_2 = np.zeros((100, 100, 3))
        self.debug_image_3 = np.zeros((100, 100, 3))

        # Read images of Cluedo characters.
        self.mustard = cv2.imread(f'{project_dir}/cluedo_images/mustard.png')
        self.peacock = cv2.imread(f'{project_dir}/cluedo_images/peacock.png')
        self.plum = cv2.imread(f'{project_dir}/cluedo_images/plum.png')
        self.scarlet = cv2.imread(f'{project_dir}/cluedo_images/scarlet.png')

        self.character_imgs = [
            self.mustard,
            self.peacock,
            self.plum,
            self.scarlet
        ]

        # Alternative methods. Left here as comment for record purposes.

        # Use CRNN based model provided by OpenCV (the onnx file i.e. the trained model weights were downloaded from
        # the google drive link provided by OpenCV documentation)
        # Based on: https://stackoverflow.com/questions/67763853/text-recognition-and-restructuring-ocr-opencv
        # self.dnn_text_recognition_model = cv2.dnn.readNet(f'{project_dir}/models/crnn.onnx')
        # self.orb = cv2.ORB_create()

        # self.characters_orb = [
        #     self.orb.detectAndCompute(self.mustard, None),
        #     self.orb.detectAndCompute(self.peacock, None),
        #     self.orb.detectAndCompute(self.plum, None),
        #     self.orb.detectAndCompute(self.scarlet, None)
        # ]

        # self.brute_force_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)


        # Will control whether debug code (logging, showing images etc.) will be executed.
        self.debug = debug

        # Image from the camera.
        self.image = None

        self.potential_character_image_hsv = None

        
        self.pub_area = rospy.Publisher('identified_area', Int64, queue_size=0)
        self.pub_center = rospy.Publisher('identified_center', Int64, queue_size=0)
        self.pub_image = rospy.Publisher('identified_character_image', Image, queue_size=0)
        self.pub_pipeline_image = rospy.Publisher('identified_character_image_pipeline', Image, queue_size=0)

        # Subscribe to the camera output from the turtlebot.
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/rgb/image_raw", Image,
                         self.callback_process_image)
        
        rospy.Subscriber("execution_shutdown", Int8,
                         self.do_shutdown)

        # Publisher to broadcast a message that a character has been identified.
        self.identified_character_publisher = rospy.Publisher(
            'identified_name', Int8, queue_size=0)
        
        self.shutdown = False
        
    def do_shutdown(self, i):
        if i.data == 42:
            self.shutdown = True

    def callback_process_image(self, data):
        """Process the received image.

            This function will process the raw image and call a function to
            detect characters.

        Args:
            data (_type_): The received image
        """

        if self.shutdown:
            return

        # Convert the received image into a opencv image
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as error:
            print(error)

        # Convert the rgb image into a hsv image
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # See if there is a character in the view.
        character_detection_result = self.detect_rectangle_using_contours(hsv_image)

        if character_detection_result != -1:
            self.identified_character_publisher.publish(character_detection_result)

        # Show the debug image if we are in debug mode.
        if self.debug:
            cv2.imshow('Debug Image', self.cv_pipeline_image)
            cv2.waitKey(1)

            cv2.imwrite(f'{self.project_dir}/debug/debug_image.png', self.cv_pipeline_image)
            cv2.imwrite(f'{self.project_dir}/debug/debug_image_1.png', self.debug_image_1)
            cv2.imwrite(f'{self.project_dir}/debug/debug_image_2.png', self.debug_image_2)
            cv2.imwrite(f'{self.project_dir}/debug/debug_image_3.png', self.debug_image_3)
        
            # Previously these were shown using cv2.imshow(), however, there were threadlock issues
            # on WSL2. A workaround is to write these images and observe them through vscode in
            # wsl2 or linux etc. VSCode updates the images inside the editor when the image file 
            # changes, which emulates the same result as cv2.imshow() but without threadlock issues
            # (threadlocking meant that gazebo would have to be restarted) 

    def detect_rectangle_using_contours(self, image=None, min_shape_factor=0.3, max_shape_factor=1.0):
        """Detects a rectangle using contours and shape-factor based method. Returns information about the detected rectangle(s).

        Args:
            :param thresholded_image: The result we get after we threshold the image.
            :param max_shape_factor:
            :param thresholded_image:
            :param min_shape_factor:
        """

        # Threshold BGR colour image using Otsu (dynamic thresholding) to seperate background and foreground.
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray_blurred = cv2.GaussianBlur(image_gray, (7, 7), 0)

        # Threshold at the 95th quantile - this will reduce impact of very bright pixels (the brick wall and floors for example).
        grayscale_threshold_value = np.quantile(np.array(image_gray_blurred).flatten(), 0.95)

        # Threshold the image. All the pixels we want are set to 255 (white).
        (_, thresholded_image_grayscale) = cv2.threshold(image_gray_blurred, grayscale_threshold_value, 255, cv2.THRESH_BINARY)

        # Carry out edge detection using Canny method on the threshold image. Apperture size is adjusted to control
        # the granularity used. 
        thresholded_image_canny = cv2.Canny(thresholded_image_grayscale, 10, 300, apertureSize=5)

        # List of rectangles to store any rectangles found.
        rectangle_list = []

        # Look for rectangles using the cv2.findContours() function. We are using cv2.RETR_EXTERNAL
        # because we just want the outside contours of the object.
        potential_contours, _ = cv2.findContours(thresholded_image_canny, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
        if len(potential_contours) > 0:
            for potential_contour in potential_contours:
                # We are expecting the rectangle to be a closed shape.
                # contour_perimeter = cv2.arcLength(
                #     potential_contour, closed=True)

                contour_area = cv2.contourArea(potential_contour)

                if contour_area <= 30.0:
                    continue

                # # Fit an rectangle, if we think the shape is an ellipsoid.
                approx_vertices = cv2.approxPolyDP(potential_contour, cv2.arcLength(potential_contour, True), True)
            
                if len(approx_vertices):
                    rectangle_list.append(potential_contour)

                # print(f'Shape factor: {shape_factor}, Contours found: {len(potential_contours)}')

        # The -1 represents no character detected.
        result = -1

        character_rectangle = None
        identified_character = None

        # Make sure we have found at least one rectangle.
        if len(rectangle_list) > 0:
            for rectangle in rectangle_list:
                (x, y, width, height) = cv2.boundingRect(rectangle)

                self.potential_character_image_hsv = cv2.cvtColor(self.image[y:y + height, x:x + width], cv2.COLOR_BGR2HSV)
                
                rectangle_area = cv2.contourArea(rectangle)


                # _, potential_character = self.orb.detectAndCompute(, None)

                # The following commented out code was used to work out the mean hue of each character (HSV colour space).
                # Character 0 mean hsv colour: (35.58036614628565, 95.93621045179718, 166.89357638817432, 0.0)
                # Character 1 mean hsv colour: (76.85818828381277, 89.68980256253009, 164.3497693064376, 0.0)
                # Character 2 mean hsv colour: (114.21065347398574, 61.854062141149065, 83.04962483999621, 0.0)
                # Character 3 mean hsv colour: (22.284440429201155, 114.16280050871943, 150.05724739980985, 0.0)

                # The values above are used to detect the character.

                mean_potential_character_hsv = np.array(cv2.mean(self.potential_character_image_hsv))

                mean_mustard_hsv = np.array([35.58036614628565, 95.93621045179718, 166.89357638817432, 0.0])
                mean_peacock_hsv = np.array([76.85818828381277, 89.68980256253009, 164.3497693064376, 0.0])
                mean_plum_hsv = np.array([114.21065347398574, 61.854062141149065, 83.04962483999621, 0.0])
                mean_scarlet_hsv = np.array([22.284440429201155, 114.16280050871943, 150.05724739980985, 0.0])

                # Was 45
                if rectangle_area >= 60:
                    # Mustard - index 0
                    if np.linalg.norm(mean_potential_character_hsv - mean_mustard_hsv, 2) <= self.sensitivity / 1.25:
                        result = 0
                        identified_character = 'Mustard'
                        character_rectangle = (x, y, width, height)

                        self.notify_character_found(cv2.cvtColor(image, cv2.COLOR_HSV2BGR))

                    # Peacock - index 1
                    elif np.linalg.norm(mean_potential_character_hsv - mean_peacock_hsv, 2) <= self.sensitivity:
                        result = 1
                        identified_character = 'Peacock'
                        character_rectangle = (x, y, width, height)

                        self.notify_character_found(cv2.cvtColor(image, cv2.COLOR_HSV2BGR))

                    # Plum - index 2
                    elif np.linalg.norm(mean_potential_character_hsv - mean_plum_hsv, 2) <= self.sensitivity * 1.2:
                        result = 2
                        identified_character = 'Plum'
                        character_rectangle = (x, y, width, height)

                        self.notify_character_found(cv2.cvtColor(image, cv2.COLOR_HSV2BGR))

                    # Scarlet - index 3
                    elif np.linalg.norm(mean_potential_character_hsv - mean_scarlet_hsv, 2) <= self.sensitivity / 1.25:
                        result = 3
                        identified_character = 'Scarlet'
                        character_rectangle = (x, y, width, height)

                        self.notify_character_found(cv2.cvtColor(image, cv2.COLOR_HSV2BGR))

                
                if identified_character is not None and contour_area >= 1.0:
                    moments = cv2.moments(potential_contour)
                    centre_x = int(moments["m10"] / moments["m00"])
                
                    self.pub_area.publish(int(contour_area))
                    self.pub_center.publish(centre_x)

                if identified_character is not None:
                    cv2.rectangle(self.image, (x, y), (x + width, y + height), (50, 50, 150), thickness=2)
                        
                ######################
                # Code
                ######################
                # for index, character in enumerate(self.character_imgs):
                #     mean_hue_character = cv2.mean(cv2.cvtColor(character, cv2.COLOR_BGR2HSV))
                    
                #     print(f'Character {index} mean hsv colour: {mean_hue_character}')

                ##########################################################################
                # Alternative Methods Attempted to Detect Characters
                ##########################################################################
                # 1) [NOT SUCCESSFUL] Feature Descriptors: An attempt was made to use OpenCV ORB (cv2.ORB_Create()) to
                # detect features in the query (grayscale) images of the characters and the detected polygons that could
                # have the frame inside them. This method was not very reliable, as scarlet was many times recognised 
                # incorrectly as other characters. This could possibly be due to the smaller camera image size (i.e.
                # scene image resolution) along with the similarity between all the images of the characters (for example text
                # in all character images is placed at the same locations, the images of characters are placed is similar
                # locations etc.)
                # 
                # 2) [NOT SUCCESSFUL] Harris Corners based attempt: There was a hypothesis that due to the many curves in the character images
                # (e.g. the curves in the text, the curves in the character faces etc.) combined with the lower resolution of the 
                # scene image, would lead to more corners being detected inside rectangular frames and not in other false 
                # positives (like the rectangular cupboard). However, after implementation this method was not successful 
                # the cupboard for example had many corners detected on the edges as well.
                # 
                # 3) [NOT SUCCESSFUL] Detect the text using Deep Learning (OpenCV's dnn module was used for this attempt and the weights 
                # were provided by OpenCV on this link (see models/ folder for the .onnx files used). Both crnn.onnx and crnn_cs.onnx
                # did not detect most letters correctly.
                # 
                # 4) [NOT SUCCESSFUL] Template matching inside the detected rectangles (not over the entire scene image) - peacock and scarlet
                # were being confused with each other (potentially due to cv2.matchTemplate requiring grayscale templates and query
                # images). A further avenue of interest would be to see if using template matching on all the three channels
                # seperately (RGB) and having the constraint of having a match with a higher probability in all channels might
                # lead to more success with this approach.

            if identified_character is not None:
                (x, y, width, height) = character_rectangle
                cv2.rectangle(self.image, (x, y), (x + width, y + height), (255, 50, 50), thickness=4)
                cv2.putText(self.image, identified_character, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 50, 50), thickness=3)
                    
        image = cv2.putText(cv2.cvtColor(image, cv2.COLOR_HSV2BGR), "Camera Image", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), thickness=2)
        image = cv2.copyMakeBorder(image,3,3,3,3,cv2.BORDER_CONSTANT,value=[127, 127, 127] )

        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        image_gray = cv2.copyMakeBorder(image_gray,3,3,3,3,cv2.BORDER_CONSTANT,value=[127, 127, 127] )
        cv2.putText(image_gray, "Grayscale Image", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), thickness=2)
        

        image_gray_blurred = cv2.cvtColor(image_gray_blurred, cv2.COLOR_GRAY2BGR)
        image_gray_blurred = cv2.copyMakeBorder(image_gray_blurred,3,3,3,3,cv2.BORDER_CONSTANT,value=[127, 127, 127] )
        cv2.putText(image_gray_blurred, "Gaussian Blur to remove noise", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), thickness=2)

        thresholded_image_grayscale = cv2.cvtColor(thresholded_image_grayscale, cv2.COLOR_GRAY2BGR)
        thresholded_image_grayscale = cv2.copyMakeBorder(thresholded_image_grayscale,3,3,3,3,cv2.BORDER_CONSTANT,value=[127, 127, 127] )
        cv2.putText(thresholded_image_grayscale, "Thresholded Image", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), thickness=2)

        thresholded_image_canny = cv2.cvtColor(thresholded_image_canny, cv2.COLOR_GRAY2BGR)
        thresholded_image_canny = cv2.copyMakeBorder(thresholded_image_canny,3,3,3,3,cv2.BORDER_CONSTANT,value=[127, 127, 127] )
        cv2.putText(thresholded_image_canny, "Canny Edge Detection", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), thickness=2)

        self.image = cv2.copyMakeBorder(self.image,3,3,3,3,cv2.BORDER_CONSTANT,value=[127, 127, 127] )
        cv2.putText(self.image, "Character Identification Visualisation", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), thickness=2)
        
        image_grid_2d = [
            [image, image_gray, image_gray_blurred],
            [thresholded_image_grayscale, thresholded_image_canny, self.image]
        ]

        concatenated_image = cv2.vconcat([cv2.hconcat(row_list) for row_list in image_grid_2d])

        self.cv_pipeline_image = concatenated_image

        return result

    def notify_character_found(self, image):
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(image, encoding="bgr8"))
        self.pub_pipeline_image.publish(self.bridge.cv2_to_imgmsg(self.cv_pipeline_image, encoding="bgr8"))
    
    def detect_text_using_deep_learning_crnn_model(self, gray_image=None):
        blob = cv2.dnn.blobFromImage(gray_image, size=(100,32))

        self.dnn_text_recognition_model.setInput(blob)

        scores = self.dnn_text_recognition_model.forward()
        print(scores.shape)

        alphabet_set = "0123456789abcdefghijklmnopqrstuvwxyz"
        blank = '-'

        charset = blank + alphabet_set

        # Decode text from model probability scores.
        detected_text = ''

        for i in range(scores.shape[0]):
            charset_index = np.argmax(scores[i][0])

            detected_text += charset[charset_index]

        return detected_text



    # Template matching?
