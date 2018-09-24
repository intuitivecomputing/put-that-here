Put That Here: Situated Multimodal Instruction of Robot Pick-and-Place Tasks
==========================================================
### Author: Ji Han, Ze Li, Chien-Ming Huang
### E-mail: jhan53@jhu.edu
****
Prerequisite
------
1. Purple gloves for both hands
2. Green tablecloth
3. Top-Down Webcam
4. UR5
5. ROS Kinetics enviroment
6. OpenCV

Installation
------------
    git clone https://github.com/intuitivecomputing/put-that-here.git

[UR5 Driver](https://github.com/intuitivecomputing/icl_phri_ur5)

[Google Speech-to-Text](https://cloud.google.com/speech-to-text/docs/reference/libraries)


Usage
----------
This project supports multimodal robot instruction with four kinds of gestures and various verbal commands.
![GitHub Logo](/image/img1.png)
### Gesture-Recognition
    rosrun put_that_here hand_tracking_node.py
### Speech-Recognition
    rosrun put_that_here Speech_node.py
### Autonomous Manipulation
    rosrun put_that_here pick_ik.py
### Wizard-of-Oz Manipulation
    rosrun put_that_here control_group.py
### Signaling Feedback
This node should be run on another computer that connected to the projector.
    
    rosrun put_that_here projector.py


Reference
----------
[Google Cloud API](https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/speech/cloud-client/transcribe_streaming_mic.py)


[python_UR5_ikSlover](https://github.com/fjonath1/python_UR5_ikSolver)
