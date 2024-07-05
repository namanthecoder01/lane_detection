Car and Lane Detection Project

Overview

This project uses OpenCV for real-time car detection and lane detection in a video stream. It combines the detection of cars using Haar cascades and lane detection using Canny edge detection and Hough transform.

Dependencies

Python 3.x
OpenCV (cv2)
numpy

Installation

Install Python (if not already installed) from python.org.

Install OpenCV:
pip install opencv-python

Install numpy:
pip install numpy

Usage

Place your video file named test.mp4 in the project directory.

Run the script:

python your_script_name.py

Press 'q' to quit the application.

Future Advancements

Object Detection in Front of Cars: Implement object detection techniques (e.g., using YOLOv3, SSD) to detect objects in front of cars, such as obstacles or road signs.

Lane Departure Warning System: Enhance the lane detection algorithm to provide warnings when the vehicle deviates from its lane.

Tyre Puncture Prevention System: Integrate sensors or image processing techniques to detect sharp objects (like pins) on the road to prevent tyre punctures.

Author
Naman Jain
