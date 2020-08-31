import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os

def hls_thresh2(img):
	# Using inRange() to threshold both white and yellow separately.
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	# White areas in image
	# H value can be arbitrary, thus within [0 ... 360] (OpenCV: [0 ... 180])
	# L value must be relatively high (we want high brightness), e.g. within [0.7 ... 1.0] (OpenCV: [0 ... 255])
	# S value must be relatively low (we want low saturation), e.g. within [0.0 ... 0.3] (OpenCV: [0 ... 255])
	white_lower = np.array([np.round(  0 / 2), np.round(0.43 * 255), np.round(0.00 * 255)])
	white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.30 * 255)])
	white_mask = cv2.inRange(hls, white_lower, white_upper)

	# Yellow areas in image
	# H value must be appropriate (see HSL color space), e.g. within [40 ... 60]
	# L value can be arbitrary (we want everything between bright and dark yellow), e.g. within [0.0 ... 1.0]
	# S value must be above some threshold (we want at least some saturation), e.g. within [0.35 ... 1.0]
	yellow_lower = np.array([np.round( 35/ 2 ), np.round(0.00 * 255), np.round(0.2 * 255)])
	yellow_upper = np.array([np.round( 65 / 2 ), np.round(1.00 * 255), np.round(1.00 * 255)])
	yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
	binary_output = cv2.bitwise_or(yellow_mask, white_mask)

	return binary_output


def combined_thresh_canny(img):
    canny = cv2.Canny(img, 20, 150)
    kernel = np.ones((3,3),np.uint8)
    canny = cv2.dilate(canny,kernel,iterations = 1)
    hls_bin = hls_thresh2(img)
    combined = np.zeros(img.shape)
    combined[(canny == 255) & (hls_bin == 255)] = 1
    

    return combined, canny,  hls_bin  # DEBUG datatype


if __name__ == '__main__':
	img_file = os.path.dirname(os.path.abspath(__file__))+'/saves/111_new.png'
	img = mpimg.imread(img_file)
	if img.dtype == 'float32':
		img = np.array(img)*255
		img = np.uint8(img)
	# with open('calibrate_camera.p', 'rb') as f:
	# save_dict = pickle.load(f)
	# mtx = save_dict['mtx']
	# dist = save_dict['dist']


	# img = cv2.undistort(img, mtx, dist, None, mtx)
	combined, canny, hls_bin = combined_thresh_canny(img)


	plt.subplot(2, 2, 1)
	plt.title("Canny")
	plt.imshow(canny, cmap='gray')

	plt.subplot(2, 2, 2)
	plt.title("Original")
	plt.imshow(img)

	plt.subplot(2, 2, 3)
	plt.title("HLS Threshold")
	plt.imshow(hls_bin, cmap='gray')

	plt.subplot(2, 2, 4)
	plt.title("Combined Threshold")
	plt.imshow(combined, cmap='gray')

	plt.tight_layout()
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	fig.canvas.set_window_title("Thresholds")
	plt.show()
	
	# #NEW
	# lines = cv2.HoughLinesP(np.uint8(combined), 2, np.pi/180, 100, np.array([]), 50, 80)
	# line_image = np.zeros((combined.shape[0], combined.shape[1], 3), dtype=np.uint8)
	# try:
	# 	for line in lines:
	# 		for x1,y1,x2,y2 in line:
	# 				cv2.line(line_image, (x1, y1), (x2, y2), [255, 0, 0], 20)
	# except TypeError:
	# 	pass
                
	# a = 1
	# b = 1
	# c = 0    
	# # Resultant weighted image is calculated as follows: original_img * a + img * b + c
	# Image_with_lines = cv2.addWeighted(img2, a, line_image, b, c)
	# plt.imshow(Image_with_lines)
	# plt.show()

	# cv2.imshow("HLS", np.uint8(binary_output))
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	cv2.destroyAllWindows()

	# while(True):
	# 	cv2.imshow("HLS", np.float64(combined))
	# 	if cv2.waitKey(1) & 0xFF == ord('q'):
	# 		cv2.destroyAllWindows()
	# 		break
