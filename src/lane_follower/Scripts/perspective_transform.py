#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from combined_thresh_canny import combined_thresh_canny

import os


def perspective_transform(img):
	"""
	Execute perspective transform
	"""
	
	img_size = (img.shape[1], img.shape[0])
	# src = np.float32(
	# 	[[115, 171], #60 -- Works for wide lane
	# 	 [525, 177],
	# 	 [0, 315], #20
	# 	 [640, 315]])
	src = np.float32(
		[[190, 210], #60 -- First time with ribbons Sep 8
		 [620-190, 210],
		 [0, 480], #20
		 [620, 480]])

	dst = np.float32(
		 [[0, 0],
		 [img.shape[1], 0],
		 [0, img.shape[0]],
		 [img.shape[1], img.shape[0]]])

	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)

	warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
	unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

	return warped, unwarped, m, m_inv


if __name__ == '__main__':
	img_file = os.path.dirname(os.path.abspath(__file__))+'/saves/64_igvcw.png' #2019-12-19-153614.jpg

	with open(os.path.dirname(os.path.abspath(__file__))+'/calibrate_camera.p', 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']

	img = mpimg.imread(img_file)
	plt.imshow(img)
	plt.show()
	if img.dtype == 'float32':
		img = np.array(img)*255
		img = np.uint8(img)
	img = cv2.blur(img, (5,5))
	img = cv2.undistort(img, mtx, dist, None, mtx)
	plt.imshow(img)
	plt.show()
	img, _, _, _, _ = combined_thresh(img)
	warped, unwarped, m, m_inv = perspective_transform(img)

	plt.subplot(3,1,1)
	plt.imshow(img, cmap='gray',)
	plt.subplot(3,1,2)
	plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3,1,3)
	plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	plt.show()
