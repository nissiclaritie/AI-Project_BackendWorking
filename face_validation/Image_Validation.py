
# import the necessary packages
from imutils import face_utils
import numpy as np

import imutils
import dlib
import cv2
import os
# function to detect the forehead , its pixels and validations
# def detect_forehead():
# 	# get upper eyes landmarks position and face rectangles
#
#
# 	pass
# # function to detect the forehead , its pixels and validations
# def detect_chin():
# 	print(rects)
# 	pass
# # function to detect the forehead , its pixels and validations
# def detect_cheek():
# 	pass




# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
path = '/home/manoj/Downloads/wiki_crop/00'
def detect_face():
	for file in os.listdir(path):
		# load the input image, resize it, and convert it to grayscale
		image = cv2.imread(os.path.join(path,file))
		# image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale image
		rects = detector(gray, 1)
		for k, d in enumerate(rects):
			print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(),
																			   d.bottom()))
			image = cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 2)
		# image = cv2.rectangle(image, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (255, 0, 0) , 1)
		return image,rects
		# cv2.imshow('image', image)
		# cv2.waitKey(0)
image ,rects = detect_face()
# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# dictionary to store the locations of each landmarks and there ROI
	dict_lm = {}



	# loop over the face parts individually
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():


		# clone the original image so we can draw on it, then
		# display the name of the face part on the image
		clone = image.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)

		# loop over the subset of facial landmarks, drawing the
		# specific face part
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

		# extract the ROI of the face region as a separate image
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		dict_lm[name] =(x, y, w, h)

		# code to cut the ROI from image
		# roi = image[y:y + h, x:x + w]
		# roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

		# show the particular face part
		# cv2.imshow("ROI", roi)
		# cv2.imshow("Image", clone)
		# cv2.waitKey(0)

	# visualize all facial landmarks with a transparent overlay
	output = face_utils.visualize_facial_landmarks(image, shape)
	image = cv2.rectangle(image, rects[0][0], rects[0][1], (255, 0, 0) , 1)
	cv2.imshow("Image", image)
	cv2.waitKey(0)

print(dict_lm)