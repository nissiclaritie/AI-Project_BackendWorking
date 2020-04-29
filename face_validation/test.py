import cv2
img = cv2.imread('/home/manoj/Downloads/office/Makeup_AI/client_Data/perfect_1.jpeg')
roi = cv2.selectROI(img, False)
cv2.imshow('t',roi)
cv2.waitKey(0)