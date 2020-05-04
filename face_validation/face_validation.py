import os
import cv2
from imutils import face_utils
import dlib
import numpy as np
import imutils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('face_validation/shape_predictor_81_face_landmarks.dat')

def verify_angle(shape,rects):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]
    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # check for angle of forehead

    # compute the center of mass for each eye
    headStart = shape[77]
    headend = shape[78]
    # compute the angle between the eye centroids
    dY = headend[1] - headStart[1]
    dX = headend[0] - headStart[0]
    headangle = np.degrees(np.arctan2(dY, dX)) - 180



    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]


    jawpt = shape[lStart:lEnd+1]

    # compute the center of mass for each eye
    jawpts = jawpt[0:int(len(jawpt)/2)].mean(axis=0).astype("int")
    jawpts2 = jawpt[int(len(jawpt)/2):].mean(axis=0).astype("int")
    # compute the angle between the eye centroids
    dY = jawpts2[1] - jawpts[1]
    dX = jawpts2[0] - jawpts[0]
    jawangle = np.degrees(np.arctan2(dY, dX)) - 180

    angle_res = {}
    # conditions to validate the angles
    if  0.0 <= abs(angle)<=2.5 or 358.0 <= abs(angle)<=360.0:
        angle_res['eye'] = 'Passed'

    else:
        angle_res['eye'] = 'Failed'

    if 179.0 <= abs(headangle)<=182.0 :
        angle_res['head'] = 'Passed'

    else:
        angle_res['head'] = 'Failed'

    if 187.0 <= abs(jawangle)<=191.0:
        angle_res['jaw'] = 'Passed'

    else:
        angle_res['jaw'] = 'Failed'


    return angle_res

def detect_face(gray,image):
    # detect rectangle and mark on face
    rects = detector(gray, 1)
    # for k, d in enumerate(rects):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(),
        #                                                                    d.bottom()))
        # image = cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 2)


    # image = cv2.rectangle(image, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (255, 0, 0) , 1)
    return image, rects


def detect_landmarks(rects,gray,image):
    # detect landmarks and mark on face
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

            dict_lm[name] = (x, y, w, h)
        output = face_utils.visualize_facial_landmarks(image, shape)
        return output,shape


def roi_face(angle,shape,image):
    if angle['head'] == 'Passed' and angle['jaw'] == 'Passed' or angle['head'] == 'Passed' and angle[
        'eye'] == 'Passed' or angle['eye'] == 'Passed' and angle['jaw'] == 'Passed':
    # print(shape)
    # (tlx, tly), (trx,tRy), (blx, bly),(brx,bry)
        (tlx, tly) = shape[70]
        (trx, tRy) = shape[80]
        (blx, bly) = shape[20]
        (brx, bry) = shape[23]

        x1 = min(tlx, trx, brx, blx)
        x2 = max(tlx, trx, brx, blx)
        y1 = min(tly,tRy, bry, bly)
        y2 = max(tly,tRy, bry, bly)
        # roi_forehead = image[x2:y2,x1:y1] #int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])
        roi_forehead = image[y1:y2,x1:x2]
        return roi_forehead
    # cv2.imshow('roiim', roi_forehead)
    # cv2.waitKey(0)


    # cut the ROI to process the skintone

    pass



def process_file(filename):
    # read image and convert to gray scale for next steps
    image = cv2.imread(filename)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image , rects = detect_face(gray,image)
    if len(rects)>0:

        outimage,shape = detect_landmarks(rects,gray,image)
        angle = verify_angle(shape,rects)
        return angle,shape,image
    else:
        return False


# folder_path = '/home/manoj/Downloads/office/Makeup_AI/data/client_Data'
# for file in os.listdir(folder_path):
#     filename = os.path.join(folder_path,file)
#     process_file(filename)
