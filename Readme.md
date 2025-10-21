# Sensor Networks and Sensor Data Fusion
## Programming Exercise 2: Face Detection and 3D Reconstruction

### Description
#### Files
* `face.py` is your main program. 
* `calibration_data.pkl` contains camera calibration data for my computer's camera.

You can run the program using `python face.py` after setting up your work environment. See below.

I have left the function `processFace()` as exercise for you. It receives the camera matrix and the coordinates of the upper left and lower right corner of the first face found in the camera picture. It should return a 3-vector with the (x,y,z) coordinates of the face's center in the camera frame. These coordinates then get plotted. Please use the information you learned in the lecture to fill the function. Basic trigonometry and the camera matrix should be sufficient.

If you have questions or comments chat me up in Teams or write an email to bjoern.giesler@thi.de!

### Set up your work environment 
To prepare your work environment, you need
* Python 3.10 or later -- www.python.org
* The DearPyGui user interface library. Install with "pip install dearpygui" (pip comes with the Python3.x you just installed).
* The OpenCV computer vision library. Install with "pip install opencv-python".
* Nothing else should be required. Let me know if you can't get it to work.

### Some usage hints
* The Graph visualization is quite powerful. 
* Double clicking auto-scales the view to show all data points.
* Pan vertically by clicking and dragging with the left mouse button.
* Zoom in or out with the mouse wheel.
* Drag with the right mouse button to zoom into a specific vertical region.
* Click individual curves in the legend (top left corner) to enable/disable.
