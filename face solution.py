"""
face solution.py

Face 3D position demo with robust calibration loading and cleanup fixes.

Usage:
    python "face solution.py"

Notes:
- Ensure calibration_data.pkl (containing 'camera_matrix' and 'distortion_coefficients')
  is in the same directory as this script, or update CALIBRATION_FILE accordingly.
- Requires: opencv-python, numpy, dearpygui
"""

from typing import Tuple
import os
import pickle
import warnings

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from DataPlot import DataPlot

# Load calibration file from script directory to avoid working-directory issues
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), 'calibration_data.pkl')

# Average face height used to estimate depth (mm)
FACE_HEIGHT_MM = 230.0


class FaceGui:
    def __init__(self):
        # Create all data plots to hold 1000 points before scrolling
        self.xyzplot = DataPlot(("x", "y", "z"), 1000)

        # Initialize face detector
        cascPathface = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_alt2.xml")
        self.faceCascade = cv2.CascadeClassifier(cascPathface)

        # Open default camera
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            raise RuntimeError("Could not open video capture device (camera).")

        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video resolution: {self.width}x{self.height}")

        # Load calibration data and initialize undistortion maps
        if not os.path.exists(CALIBRATION_FILE):
            raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_FILE}")

        with open(CALIBRATION_FILE, 'rb') as f:
            calibration_data = pickle.load(f)

        if 'camera_matrix' not in calibration_data or 'distortion_coefficients' not in calibration_data:
            raise KeyError("calibration_data.pkl must contain 'camera_matrix' and 'distortion_coefficients' keys.")

        mtx = np.asarray(calibration_data['camera_matrix'], dtype=float)
        dist = np.asarray(calibration_data['distortion_coefficients'], dtype=float)
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients:\n{dist}")

        # Use optimal new camera matrix for undistortion and compute maps
        self.mtx, _roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.width, self.height), 1, (self.width, self.height))
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, self.mtx, (self.width, self.height), cv2.CV_32FC1)

    def createWindow(self):
        with dpg.window(tag="Status"):
            with dpg.group(horizontal=True):
                self.xyzplot.createGUI(-1, -1)

    def processFace(self, K: np.ndarray, u0: float, v0: float, u1: float, v1: float) -> Tuple[float, float, float]:
        """
        Estimate 3D position (x, y, z) of a face center in the CAMERA coordinate frame.

        Inputs
        ------
        K : (3x3) numpy array
            Camera intrinsic matrix
        u0, v0 : float
            Top-left pixel coordinates of the face bounding box
        u1, v1 : float
            Bottom-right pixel coordinates of the face bounding box

        Returns
        -------
        (x_mm, y_mm, z_mm) : tuple of floats
            3D coordinates in millimetres in the camera coordinate frame.
            Z increases with distance from the camera. X to the right, Y down.
        """
        # Small epsilon to avoid division by zero
        _eps = 1e-8

        # Convert inputs to floats for stable math
        u0 = float(u0); v0 = float(v0); u1 = float(u1); v1 = float(v1)

        # Height of bounding box in pixels (use height to estimate distance)
        h_pixels = (v1 - v0)
        if not np.isfinite(h_pixels) or h_pixels <= 0:
            # Invalid bbox height -> cannot estimate depth
            return 0.0, 0.0, 0.0

        # Validate K
        K = np.asarray(K, dtype=float)
        if K.shape != (3, 3):
            return 0.0, 0.0, 0.0

        # Prefer fy for vertical size based depth estimate; fallback to fx
        fy = K[1, 1]
        if not np.isfinite(fy) or abs(fy) < _eps:
            fy = K[0, 0]
            if not np.isfinite(fy) or abs(fy) < _eps:
                return 0.0, 0.0, 0.0

        # Pinhole geometry: h_pixels ≈ (FACE_HEIGHT_MM * fy_pixels) / Z_mm
        # => Z_mm = FACE_HEIGHT_MM * fy / h_pixels
        z_mm = (FACE_HEIGHT_MM * fy) / (h_pixels + _eps)

        # Center pixel of bounding box
        u_c = (u0 + u1) / 2.0
        v_c = (v0 + v1) / 2.0

        # Back-project: [X, Y, Z]^T = Z * K^{-1} * [u, v, 1]^T
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)

        uv1 = np.array([u_c, v_c, 1.0], dtype=float)
        norm_dir = K_inv.dot(uv1)  # [X/Z, Y/Z, 1]^T

        x_mm = float(norm_dir[0] * z_mm)
        y_mm = float(norm_dir[1] * z_mm)
        z_mm = float(z_mm)

        if not (np.isfinite(x_mm) and np.isfinite(y_mm) and np.isfinite(z_mm)):
            return 0.0, 0.0, 0.0

        return x_mm, y_mm, z_mm

    def run(self):
        dpg.create_context()
        dpg.create_viewport()
        self.createWindow()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Status", True)
        frameno = 0

        try:
            while dpg.is_dearpygui_running():
                # Read frame from camera
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Undistort frame
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

                # Convert to greyscale and detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    u0, v0, u1, v1 = x, y, x + w, y + h
                    xmm, ymm, zmm = self.processFace(self.mtx, u0, v0, u1, v1)
                    self.xyzplot.addDataVector(frameno, (xmm, ymm, zmm))
                    cv2.putText(
                        frame,
                        f"x={xmm:.0f}mm y={ymm:.0f}mm z={zmm:.0f}mm",
                        (u0, v0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )
                    frameno += 1

                cv2.imshow('Video', frame)

                # Allow OpenCV window to process events (press 'q' to quit)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                dpg.render_dearpygui_frame()

        finally:
            # Cleanup section
            try:
                if hasattr(self, "video_capture") and self.video_capture is not None:
                    self.video_capture.release()
            except Exception:
                pass

            cv2.destroyAllWindows()
            dpg.destroy_context()


if __name__ == "__main__":
    gui = FaceGui()
    gui.run()
