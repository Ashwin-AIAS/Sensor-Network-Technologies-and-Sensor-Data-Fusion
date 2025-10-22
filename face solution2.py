"""
face solution.py

Corrected/merged version: uses the professor's geometry (face width) to estimate depth,
keeps robust calibration loading and cleanup, and fixes various bugs (cleanup, indentation).
"""

import os
import math
import pickle
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from DataPlot import DataPlot

# Calibration file (assumed next to this script)
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration_data.pkl")

# Use the professor's constants (face width used in their solution)
FACE_WIDTH_MM = 140  # Average width of a human face in mm (professor's choice)
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


class FaceGui:
    def __init__(self):
        # Create all data plots to hold 1000 points before scrolling
        self.xyzplot = DataPlot(("x", "y", "z"), 1000)

        # Initialize face detector (Haar cascade)
        cascPathface = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_alt2.xml")
        self.faceCascade = cv2.CascadeClassifier(cascPathface)

        # Open default camera
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            raise RuntimeError("Cannot open camera.")

        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video resolution: {self.width}x{self.height}")

        # Load calibration
        if not os.path.exists(CALIBRATION_FILE):
            raise FileNotFoundError(f"Missing calibration file: {CALIBRATION_FILE}")

        with open(CALIBRATION_FILE, "rb") as f:
            calibration_data = pickle.load(f)

        if "camera_matrix" not in calibration_data or "distortion_coefficients" not in calibration_data:
            raise KeyError("calibration_data.pkl must contain 'camera_matrix' and 'distortion_coefficients'")

        mtx = np.asarray(calibration_data["camera_matrix"], dtype=float)
        dist = np.asarray(calibration_data["distortion_coefficients"], dtype=float)
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients:\n{dist}")

        # Use optimal new camera matrix for undistortion and compute maps
        self.mtx, _roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.width, self.height), 1, (self.width, self.height))
        # mapx/mapy as float maps
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, self.mtx, (self.width, self.height), cv2.CV_32FC1)

    def createWindow(self):
        with dpg.window(tag="Status"):
            with dpg.group(horizontal=True):
                self.xyzplot.createGUI(-1, -1)

    def processFace(self, K, u0, v0, u1, v1):
        """
        Compute (x, y, z) in mm of face center in camera coordinates using professor's approach.

        Inputs:
            K: 3x3 camera matrix (intrinsics)
            (u0, v0): top-left pixel of bounding box
            (u1, v1): bottom-right pixel of bounding box

        Returns:
            (x_mm, y_mm, z_mm)
        """
        # Validate K shape
        K = np.asarray(K, dtype=float)
        if K.shape != (3, 3):
            return 0.0, 0.0, 0.0

        # Extract intrinsics
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # Convert bbox coords to centered coords (principal point at (0,0))
        u0c = float(u0) - cx
        u1c = float(u1) - cx
        v0c = float(v0) - cy
        v1c = float(v1) - cy

        # Compute angles of left and right edges (horizontal angular positions)
        # Use atan2(u, f) where u is x in camera pixels relative to principal point
        angle1 = math.atan2(u0c, fx)
        angle2 = math.atan2(u1c, fx)

        # Avoid pathological cases where angles are equal (zero width)
        denom = math.sin(angle2) - math.sin(angle1)
        if abs(denom) < 1e-8:
            return 0.0, 0.0, 0.0

        # Depth (z) from geometry derived by professor:
        # z = face_real_width / (sin(angle_right) - sin(angle_left))
        z_mm = FACE_WIDTH_MM / denom

        # center of bbox in centered coords
        u_c = (u0c + u1c) / 2.0
        v_c = (v0c + v1c) / 2.0

        # Back-project to get X and Y (X to right, Y up in camera coordinates normally;
        # professor inverts Y to match image coords -> keep that sign convention)
        x_mm = (z_mm / fx) * u_c
        # In professor's code Y is inverted: y = -(z/fy) * v  (image v increases downward)
        y_mm = -(z_mm / fy) * v_c

        return float(x_mm), float(y_mm), float(z_mm)

    def run(self):
        """Main loop"""
        dpg.create_context()
        dpg.create_viewport()
        self.createWindow()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Status", True)
        frameno = 0

        try:
            while dpg.is_dearpygui_running():
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Resize to target resolution (professor's implementation)
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

                # Undistort
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
                )

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    u0 = x
                    v0 = y
                    u1 = x + w
                    v1 = y + h

                    # Use the post-undistortion intrinsics (self.mtx) for backprojection/angles
                    xmm, ymm, zmm = self.processFace(self.mtx, u0, v0, u1, v1)

                    self.xyzplot.addDataVector(frameno, (xmm, ymm, zmm))
                    cv2.putText(frame, f"x={xmm:.0f}mm y={ymm:.0f}mm z={zmm:.0f}mm", (u0, v0 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    frameno += 1

                cv2.imshow("Video", frame)

                # allow window to process events; press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                dpg.render_dearpygui_frame()

        finally:
            # cleanup
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
