import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from line_fit import line_fit, viz2, calc_curve, final_viz

with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

##image_files = os.listdir('test_images')
image_files=['2019-12-19-153614.jpg']
for image_file in image_files:
	out_image_file = image_file.split('.')[0] + '.png'  # write to png format
	img = mpimg.imread('test_images/' + image_file)
	plt.imshow(img)
	plt.show()
	plt.figure()

	# Undistort image
	img = cv2.undistort(img, mtx, dist, None, mtx)
	plt.imshow(img)
	plt.show()
	plt.figure()

	img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)
	plt.imshow(img, cmap='gray', vmin=0, vmax=1)
	plt.show()
	plt.figure()

	img, binary_unwarped, m, m_inv = perspective_transform(img)
	plt.imshow(img, cmap='gray', vmin=0, vmax=1)

	plt.show()
	plt.figure()

	ret = line_fit(img)
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']
	save_file = 'example_images/polyfit1_' + out_image_file
	img2=viz2(img, ret, save_file=save_file)
	
	plt.imshow(img2)
	plt.show()
	plt.figure()

	# Do full annotation on original image
	# Code is the same as in 'line_fit_video.py'
	orig = mpimg.imread('test_images/' + image_file)
	undist = cv2.undistort(orig, mtx, dist, None, mtx)
	left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

	bottom_y = undist.shape[0] - 1
	bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
	bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
	vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

	xm_per_pix = 3.2/640 # meters per pixel in x dimension
	vehicle_offset *= xm_per_pix

	img = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)
	plt.imshow(img)
	plt.show()
	plt.figure()
	plt.savefig('example_images/annotated2_' + out_image_file)

	break

	
	
