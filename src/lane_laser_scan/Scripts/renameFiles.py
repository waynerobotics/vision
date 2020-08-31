import os

#print(os.path.realpath(__file__))
#directory = os.fsencode("/home/ringo/workspace/laneDetection/vision_igvc/camera_cal/")
#print(directory)

files=os.listdir("/home/ringo/workspace/laneDetection/vision_igvc/camera_cal/")
for i in range(len(files)):
    newName = "calibration"+str(i+1)
    os.rename("camera_cal/"+files[i], "camera_cal/"+newName)
