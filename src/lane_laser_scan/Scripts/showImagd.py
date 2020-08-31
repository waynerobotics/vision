import cv2
#true for 2,6,7,9,11,12
img=cv2.imread('/home/ringo/workspace/laneDetection/vision_igvc/camera_cal/calibration1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (6,5), None)
cv2.imshow('window',gray)
cv2.waitKey()
cv2.destroyAllWindows()
