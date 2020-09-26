#!/usr/bin/env python3
from cv2 import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
from combined_thresh import combined_thresh
from combined_thresh_canny import combined_thresh_canny
from perspective_transform import perspective_transform
from line_fit import line_fit, viz2, calc_curve, final_viz
import numpy as np
import os

with open(os.path.dirname(os.path.abspath(__file__))+'/calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

#image_files = os.listdir(os.path.dirname(os.path.abspath(__file__))+"/saves/" )
#image_files=['2019-12-19-153614.jpg']
image_files=['391_igvcw.png']
for image_file in image_files:
	out_image_file = image_file.split('.')[0] + '.png'  # write to png format
	image_file=(os.path.dirname(os.path.abspath(__file__))+'/saves/' + image_file)
	
	img = mpimg.imread(image_file)
	if img.dtype == 'float32':
		img = np.array(img)*255
		img = np.uint8(img)
	img = cv2.blur(img, (5,5))
	#img = cv2.undistort(img, mtx, dist, None, mtx)
	img2, abs_bin, mag_bin, dir_bin, hls_bin= combined_thresh(img)
	#img, _, img2 = combined_thresh_canny(img)
	img3, binary_unwarped, m, m_inv = perspective_transform(img2)
	
	ret = line_fit(img3, viz=1)
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']
	save_file = os.path.dirname(os.path.abspath(__file__))+'/saves/polyfit1_' + out_image_file
	img4=viz2(img3, ret, save_file=save_file)

	# Do full annotation on original image
	# Code is the same as in 'line_fit_video.py'
	undist = img
	left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

	bottom_y = undist.shape[0] - 1
	bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
	bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
	vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

	xm_per_pix = 2.05/490 # meters per pixel in x dimension
	vehicle_offset *= xm_per_pix

	img5 = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)
	
	plt.subplot(2,3,1)
	plt.imshow(img)
	plt.title("Source Image")

	plt.subplot(2,3,2)
	plt.imshow(img2, cmap='gray', vmin=0, vmax=1)
	plt.title("Thresholded")

	plt.subplot(2,3,3)
	plt.imshow(binary_unwarped, cmap='gray')
	plt.title("ROI mask")

	plt.subplot(2,3,4)
	plt.imshow(img3, cmap='gray')
	plt.title("Inverse Perspective Transform")

	plt.subplot(2,3,5)
	plt.imshow(img4)
	plt.title("Fit lanes using sliding window search")

	plt.subplot(2,3,6)
	plt.imshow(img5)
	plt.title("Lane overlay on source")

	#plt.tight_layout()
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	fig.canvas.set_window_title(image_file)
	plt.show()
	
	#plt.savefig(os.path.dirname(os.path.abspath(__file__))+'/example_images/annotated2_' + out_image_file)

	

	
	
