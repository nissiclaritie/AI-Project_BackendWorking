# import imutils
# import cv2
# import numpy as np
#
# # Open template and get canny
# template1 = cv2.imread('IMG_20200512_1827571.jpg')
# template = cv2.resize(template1, (480, 300))
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = cv2.Canny(template, 10, 25)
# (height, width) = template.shape[:2]
# # open the main image and convert it to gray scale image
# main_image = cv2.imread('IMG_20200513_150758.jpg')
# gray_image1 = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
# gray_image = cv2.resize(gray_image1, (720, 1080))
# temp_found = None
# for scale in np.linspace(0.2, 1.0, 20)[::-1]:
#     # resize the image and store the ratio
#     resized_img = imutils.resize(gray_image, width=int(gray_image.shape[1] * scale))
#     ratio = gray_image.shape[1] / float(resized_img.shape[1])
#     if resized_img.shape[0] < height or resized_img.shape[1] < width:
#         break
#     # Convert to edged image for checking
#     e = cv2.Canny(resized_img, 10, 25)
#     match = cv2.matchTemplate(e, template, cv2.TM_CCOEFF)
#     (_, val_max, _, loc_max) = cv2.minMaxLoc(match)
#     if temp_found is None or val_max > temp_found[0]:
#         temp_found = (val_max, loc_max, ratio)
# # Get information from temp_found to compute x,y coordinate
# (_, loc_max, r) = temp_found
# (x_start, y_start) = (int(loc_max[0]), int(loc_max[1]))
# (x_end, y_end) = (int((loc_max[0] + width)), int((loc_max[1] + height)))
# # Draw rectangle around the template
# cv2.rectangle(main_image, (x_start, y_start), (x_end, y_end), (153, 22, 0), 5)
# main_image1 = cv2.resize(main_image, (300, 300))
# cv2.imshow('Template Found', main_image1)
# cv2.waitKey(0)
import numpy as np





# import numpy as np
# import cv2
#
# green = np.uint8([[[255, 255, 255]]]) #here insert the bgr values which you want to convert to hsv
# hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
# print(hsvGreen)
#
# lowerLimit = hsvGreen[0][0][0] - 10, 100, 100
# upperLimit = hsvGreen[0][0][0] + 10, 255, 255
#
# print(upperLimit)
# print(lowerLimit)












# import cv2
# import imutils
#
# image = cv2.imread("../data/data/1589372575830.JPEG") # path = path to your file
# bin = cv2.inRange(image, (255, 255, 255), (255, 255,255))
# cv2.bitwise_not(bin, bin)
# cnts = cv2.findContours(bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
# rect = cv2.boundingRect(cnts[0])
# cv2.rectangle(image, rect, (0,255,0), 1)
# cv2.imshow("image", cv2.resize(image, (320,640)))
# cv2.waitKey(0)

# (10, 255, 255)
# (-10, 100, 100)
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # frame = cv2.imread("../data/data/1589372575830.JPEG")
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_gray = np.array([0, 5, 50], np.uint8)
    upper_gray = np.array([179, 50, 255], np.uint8)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange (hsv, lower_gray, upper_gray)
    img_res = cv2.bitwise_and(frame, frame, mask=mask)
    bluecnts = cv2.findContours(mask.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(bluecnts)>0:
        blue_area = max(bluecnts, key=cv2.contourArea)
        (xg,yg,wg,hg) = cv2.boundingRect(blue_area)
        cv2.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(0,255,0),2)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('idk',img_res)

    k = cv2.waitKey(5)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()