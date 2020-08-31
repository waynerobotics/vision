import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def calibrate_camera():
	# Mapping each calibration image to number of checkerboard corners
	# Everything is (9,6) for now
	objp_dict = {
		1: (10, 7),
		2: (10, 7),
		3: (10, 7),
		4: (10, 7),
		5: (10, 7),
		6: (10, 7),
		7: (10, 7),
		8: (10, 7),
		9: (10, 7),
		10: (10, 7),
		11: (10, 7),
		12: (10, 7),
		13: (10, 7),
		14: (10, 7),
		15: (10, 7),
		16: (10, 7),
		17: (10, 7),
		18: (10, 7),
	}
	#for i in objp_dict:
		#objp_dict[i] = (10, 7)
	# List of object points and corners for calibration
	objp_list = []
	corners_list = []

	# Go through all images and find corners
	for k in objp_dict:
		nx, ny = objp_dict[k]
		##nx+= 1; ny+= 1

		# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((nx*ny, 3), np.float32)
		objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

		# Make a list of calibration images
		fname = 'camera_cal/calibration%s' % str(k)
		img = cv2.imread(fname)

		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

		# If found, save & draw corners
		if ret == True:
			# Save object points and corresponding corners
			objp_list.append(objp)
			corners_list.append(corners)

			# Draw and display the corners
			'''cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
			plt.imshow(img)
			plt.show()
			cv2.imshow(str(k),img)
			print('Found corners for %s' % fname)
			print(k)'''
		else:
			print('Warning: ret = %s for %s' % (ret, fname))

	# Calibrate camera and undistort a test image
	img = cv2.imread('camera_cal/calibration1')
	img_size = (img.shape[1], img.shape[0])
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_list, corners_list, img_size, None, None)
	return mtx, dist


if __name__ == '__main__':
	mtx, dist = calibrate_camera()
	save_dict = {'mtx': mtx, 'dist': dist}
	with open('calibrate_camera.p', 'wb') as f:
		pickle.dump(save_dict, f)
	# Undistort example calibration image
	img = mpimg.imread('camera_cal/calibration7')
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	plt.imshow(dst)
	plt.show()
	plt.savefig('test_images/undistort_calibration_new.png')
