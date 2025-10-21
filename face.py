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
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), 'calibration_data.pkl')
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
		# K is camera matrix
		# (u0,v0) is top-left of face bounding box
		# (u1,v1) is bottom-right of face bounding box
		# Returns (x,y,z) in mm of face center in camera coordinates
		# YOUR JOB: Implement this function

		return (0, 0, 0)

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