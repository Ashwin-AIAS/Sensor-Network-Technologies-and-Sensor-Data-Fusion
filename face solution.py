import dearpygui.dearpygui as dpg
import sys
import select
import time
import math
import sys
import cv2
import os
import pickle
import numpy as np

# YOUR JOB: Implement processFace to compute x,y,z from face bounding box

from DataPlot import DataPlot
CALIBRATION_FILE = 'calibration_data.pkl'
FACE_WIDTH_MM = 140  # Average width of a human face in mm

class FaceGui:
	def __init__(self):
		# Create all data plots to hold 1000 points before scrolling
		self.xyzplot = DataPlot(("x", "y", "z"), 1000)

		# Initialize face detector
		cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
		self.faceCascade = cv2.CascadeClassifier(cascPathface)
		self.video_capture = cv2.VideoCapture(0)
		self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		print(f"Video resolution: {self.width}x{self.height}")

		# Load calibration data and initialize undistortion maps
		with open(CALIBRATION_FILE, 'rb') as f:
			calibration_data = pickle.load(f)

		mtx = calibration_data['camera_matrix']
		dist = calibration_data['distortion_coefficients']
		print(f"Camera matrix:\n{mtx}")
		print(f"Distortion coefficients:\n{dist}")
		self.mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.width, self.height), 1, (self.width, self.height))
		self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, self.mtx, (self.width, self.height), 5)    

	def createWindow(self):
		with dpg.window(tag="Status"):
			with dpg.group(horizontal=True):
				self.xyzplot.createGUI(-1, -1)

	def processFace(self, K, u0, v0, u1, v1):
		"""
		Estimate 3D position (x, y, z) of a face center in the CAMERA coordinate frame.

		Inputs
		------
		K : (3x3) numpy array
			Camera intrinsic matrix
		u0, v0 : int/float
			Top-left pixel coordinates of the face bounding box
		u1, v1 : int/float
			Bottom-right pixel coordinates of the face bounding box

		Returns
		-------
		(x_mm, y_mm, z_mm) : tuple of floats
			3D coordinates in millimetres in the camera coordinate frame.
			Z increases with distance from the camera. X to the right, Y down (camera convention).
		"""
		# Robust constants and safety
		FACE_HEIGHT_MM = 230.0  # use the average face height (mm) as specified in the assignment
		_eps = 1e-6

		# Convert inputs to floats for stable math
		u0 = float(u0); v0 = float(v0); u1 = float(u1); v1 = float(v1)
		# image height of bounding box in pixels
		h_pixels = (v1 - v0)

		# Basic validation
		if h_pixels <= 0:
			# invalid bbox height -> cannot estimate depth
			# return zeros (caller visualizer expects numeric values)
			# Could also raise an exception, but returning zeros is safer for real-time loop
			#print("Warning: non-positive bounding-box height:", h_pixels)
			return (0.0, 0.0, 0.0)

		# extract focal length in pixels from camera matrix
		# K assumed of form [[fx, s, cx], [0, fy, cy], [0, 0, 1]]
		K = np.asarray(K, dtype=float)
		if K.shape != (3, 3):
			#print("Warning: K is not 3x3")
			return (0.0, 0.0, 0.0)

		fy = K[1, 1]
		# Protect against degenerate intrinsics
		if abs(fy) < _eps:
			# fallback: try fx
			fy = K[0, 0]
			if abs(fy) < _eps:
				#print("Warning: focal length is zero in K")
				return (0.0, 0.0, 0.0)

		# Pinhole model: h_pixels = (FACE_HEIGHT_MM * fy_pixels) / Z_mm  => Z_mm = FACE_HEIGHT_MM * fy / h_pixels
		z_mm = (FACE_HEIGHT_MM * fy) / (h_pixels + _eps)  # add small eps to avoid div-by-zero

		# image center (pixel) of the face bounding box
		u_c = (u0 + u1) / 2.0
		v_c = (v0 + v1) / 2.0

		# back-project to camera coordinates: X = (u - cx) * Z / fx  ; Y = (v - cy) * Z / fy
		# but do it with full inverse of K for correctness (handles skew and non-square fx/fy)
		try:
			K_inv = np.linalg.inv(K)
		except np.linalg.LinAlgError:
			# numeric fallback
			K_inv = np.linalg.pinv(K)

		uv1 = np.array([u_c, v_c, 1.0], dtype=float)
		# the homogeneous camera coordinates (X/Z, Y/Z, 1) = K_inv * [u; v; 1]
		norm_dir = K_inv.dot(uv1)  # this is (X/Z, Y/Z, 1)
		# scale by Z to get actual coordinates in mm
		x_mm = norm_dir[0] * z_mm
		y_mm = norm_dir[1] * z_mm

		# return as plain Python floats
		return (float(x_mm), float(y_mm), float(z_mm))


	def run(self):
		dpg.create_context()
		dpg.create_viewport()
		self.createWindow()
		dpg.setup_dearpygui()
		dpg.show_viewport()
		dpg.set_primary_window("Status", True)
		frameno = 0

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
			faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
			
			if len(faces) > 0:
				(x, y, w, h) = faces[0]
				cv2.rectangle(frame, (x, y), (x + w, y + h),(255,0,0), 2)
				u0 = x
				v0 = y
				u1 = x + w
				v1 = y + h
				(xmm, ymm, zmm) = self.processFace(self.mtx, u0, v0, u1, v1)
				self.xyzplot.addDataVector(frameno, (xmm, ymm, zmm))
				cv2.putText(frame, f"x={xmm:.0f}mm y={ymm:.0f}mm z={zmm:.0f}mm", (u0, v0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
				frameno = frameno + 1
			cv2.imshow('Video', frame)

			dpg.render_dearpygui_frame()
		video_capture.release()
		cv2.destroyAllWindows()
		dpg.destroy_context()


if __name__ == "__main__":
	gui = FaceGui()
	gui.run()