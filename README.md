Put That Here: Situated Multimodal Instruction of Robot Pick-and-Place Tasks
==========================================================
### Author: Ji Han, Ze Li, Chien-ming Huang
### E-mail: jhan53@jhu.edu
****
Description
------
### constant.py
contain HSV threshold for color-based segmentation.
### hand_tracking.py
Contour-based hand detection and feature extraction.
### hand_tracking_node.py
ROS node for gesture recognition. The main function is the core logic of this project.
### control_group.py
ROS node for Wizard-of-Oz method.
### pick_ik.py
ROS node for coordinate transformation and Inverse Kinematic manipulation.
### projector.py
ROS node for signal feedback, in OpenCV format.
### Intro.py
Natural behavior signal guidance.
### Speech_node.py
ROS node for speech recognition.
### model
CNN model for learning-based gesture recognition.
### fake_publisher.py
Debugging tool.
### utils.py
Utility classes and functions.


Reference
----------
[Google Cloud API](https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/speech/cloud-client/transcribe_streaming_mic.py)


[python_UR5_ikSlover](https://github.com/fjonath1/python_UR5_ikSolver)
