Face Detection and 3D Reconstruction

This project detects faces from a webcam feed and estimates their 3D position in the camera coordinate frame using a calibrated monocular camera.

## How It Works
- The function `processFace()` estimates (x, y, z) in millimetres using the pinhole camera model:
  - Depth: \( Z = f_y \times H_{real} / h_{pixels} \)
  - Back-projection: \( [X, Y, Z]^T = Z \cdot K^{-1} [u, v, 1]^T \)
- The results are plotted live using DearPyGui.

## Run
```bash
python "face solution.py"

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
