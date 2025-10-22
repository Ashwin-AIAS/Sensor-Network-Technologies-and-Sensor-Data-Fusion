import numpy as np

# Mock intrinsic parameters (px units)
fx = 800.0
fy = 800.0
cx = 320.0
cy = 240.0
K = np.array([[fx, 0.0, cx],
			  [0.0, fy, cy],
			  [0.0, 0.0, 1.0]])

# Example bounding box in pixels
u0, v0 = 200, 150   # top-left
u1, v1 = 280, 350   # bottom-right -> height = 200 px

# Create a dummy object to call the method (or call the function body separately)
class Dummy:
	def processFace(self, K, u0, v0, u1, v1):
		# paste the function body here or import the containing class and use it
		return processFace_impl(K, u0, v0, u1, v1)

# For quick testing, we can directly call the logic as a standalone (refactor if needed)
def processFace_impl(K, u0, v0, u1, v1):
	FACE_HEIGHT_MM = 230.0
	_eps = 1e-6
	u0 = float(u0); v0 = float(v0); u1 = float(u1); v1 = float(v1)
	h_pixels = (v1 - v0)
	if h_pixels <= 0:
		return (0.0, 0.0, 0.0)
	K = np.asarray(K, dtype=float)
	fy = K[1, 1]
	if abs(fy) < _eps:
		fy = K[0, 0]
		if abs(fy) < _eps:
			return (0.0, 0.0, 0.0)
	z_mm = (FACE_HEIGHT_MM * fy) / (h_pixels + _eps)
	u_c = (u0 + u1) / 2.0
	v_c = (v0 + v1) / 2.0
	try:
		K_inv = np.linalg.inv(K)
	except np.linalg.LinAlgError:
		K_inv = np.linalg.pinv(K)
	uv1 = np.array([u_c, v_c, 1.0], dtype=float)
	norm_dir = K_inv.dot(uv1)
	x_mm = norm_dir[0] * z_mm
	y_mm = norm_dir[1] * z_mm
	return (float(x_mm), float(y_mm), float(z_mm))

# Run the test
xmm, ymm, zmm = processFace_impl(K, u0, v0, u1, v1)
print(f"Estimated face center: x={xmm:.1f} mm, y={ymm:.1f} mm, z={zmm:.1f} mm")
