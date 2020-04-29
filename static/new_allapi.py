import signal
from flask import request, Flask, jsonify
from datetime import datetime
import imutils
import traceback
import cv2
import numpy as np
import mysql.connector
from multiprocessing import Process
import psutil
import math
from sklearn import neighbors
import io
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import json
import os
import base64
import logging
import time
from image import imgPred
from PIL import Image
from openalpr import Alpr
from base64 import b64decode
import imghdr

app = Flask(__name__)

logging.basicConfig(filename="test_exception.log", level=logging.DEBUG, filemode='a+',
                    format='%(asctime)s : %(pathname)s : %(lineno)d : %(message)s')
imgPred = imgPred()
my_dictionary = {}
alpr = None
# db_conn_COLOSQL = mysql.connector.connect(host="localhost", user="root", passwd="root", db="zm")
# cur_COLOSQL = db_conn_COLOSQL.cursor(dictionary=True, buffered = True)
db_conn_COLOSQL = mysql.connector.connect(host="colosql", user="zm2", passwd="FOSnNPiG2EFQQcCf", db="zm2")
cur_COLOSQL = db_conn_COLOSQL.cursor(dictionary=True, buffered=True)
LABELS = open("coco.names").read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
(W, H) = (None, None)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'JPEG'])
UPLOAD_FOLDER = "static/uploads"
# p = None
temp_data = {}
alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")
alpr.set_top_n(1)
alpr.set_default_region("US")
alpr.set_detect_region(False)
number_plate = None
user_data = {}


def insert_into_exception(short_message, error_message):
    try:
        query = """Insert into  ai_system_log (short_message, error_message) values   ('%s','%s');""" % (
            short_message, error_message)
        # print(query)
        cur_COLOSQL.execute(query)
        db_conn_COLOSQL.commit()
    except:
        db_conn_COLOSQL.rollback()
        logging.debug("error in insert into ai_system_log" + str(traceback.format_exc()))
    try:
        query = """select * from ai_system_log where short_message = '%s' order by id DESC""" % (short_message)
        cur_COLOSQL.execute(query, multi=True)
        rows = cur_COLOSQL.fetchall()
        for data in rows:
            log_id = data['id']
            file_object = open("ai_id.txt", "a")
            file_object.write(log_id)
    except:
        db_conn_COLOSQL.rollback()
        logging.debug("error in fetching data from ai_log" + str(traceback.format_exc()))


def stop_process(req_data, user_id):
    try:
        file_path = str(user_id) + ".txt"
        f = open(file_path, "w+")
        if os.stat(file_path).st_size is not 0:
            # global p
            # if p is not None:
            # if p.is_alive():
            p_id = f.read()
            # print("the process\n\n\n\n\n was terminated and a new process was started")
            # print("\n\n\n")
            for i in range(len(req_data['data'])):
                try:
                    # print("I AM HERE")
                    query = """select * from manual_query where building_id = '%s' and camera_id = '%s' and start_time = '%s' and end_time = '%s' and is_processed = 0;""" % (
                        req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                        req_data['data'][i]['start_time'],
                        req_data['data'][i]['end_time'])
                    cur_COLOSQL.execute(query, multi=True)
                    manualquery_data = cur_COLOSQL.fetchall()
                    for data in manualquery_data:
                        queryUpdt = """UPDATE manual_query SET manual_stop=1 WHERE id=%s""" % data['id']
                        #print(queryUpdt)
                        cur_COLOSQL.execute(queryUpdt, multi=True)
                        db_conn_COLOSQL.commit()
                        #print("update done")
                        time.sleep(0.5)
                except:
                    print(traceback.format_exc())
                    logging.debug("error in updating" + str(traceback.format_exc()))
                    db_conn_COLOSQL.rollback()
            os.kill(int(p_id), signal.SIGTERM)
    except:
        logging.debug("Error in stopping the process" + str(traceback.format_exc()))
        insert_into_exception("error in stopping the process", str(traceback.format_exc()))


@app.route('/stop_process', methods=['POST'])
def process_stop():
    try:
        try:
            data = request.get_json()
            print(data)
            user_id = data['user_id']
            req_data = user_data[str(user_id)]
        except:
            logging.debug("error loading data" + str(traceback.format_exc()))
        try:
            file_path = str(user_id) + ".txt"
            f = open(file_path, "w+")
            if os.stat(file_path).st_size is not 0:
                # global p
                # if p is not None:
                # if p.is_alive():
                p_id = f.read()
                # print("the process\n\n\n\n\n was terminated and a new process was started")
                # print("\n\n\n")
                for i in range(len(req_data['data'])):
                    try:
                        # print("I AM HERE")
                        query = """select * from manual_query where building_id = '%s' and camera_id = '%s' and start_time = '%s' and end_time = '%s' and is_processed = 0;""" % (
                            req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                            req_data['data'][i]['start_time'],
                            req_data['data'][i]['end_time'])
                        cur_COLOSQL.execute(query, multi=True)
                        manualquery_data = cur_COLOSQL.fetchall()
                        #print(query)
                        for data in manualquery_data:
                            queryUpdt = """UPDATE manual_query SET manual_stop=1 WHERE id=%s""" % data['id']
                            #print(queryUpdt)
                            cur_COLOSQL.execute(queryUpdt, multi=True)
                            db_conn_COLOSQL.commit()
                            #print("update done")
                            time.sleep(0.5)
                    except:
                        print(traceback.format_exc())
                        logging.debug("error in updating" + str(traceback.format_exc()))
                        db_conn_COLOSQL.rollback()
                os.kill(int(p_id), signal.SIGTERM)
                return "Done"
        except:
            logging.debug("Error in stopping the process" + str(traceback.format_exc()))
    except:
        insert_into_exception("error in stop process api", str(traceback.format_exc()))
        return "False"

@app.route('/box_motion_check', methods=['POST'])
def receive_and_response():
    try:
        # global p
        global temp_data
        try:
            req_data = request.get_json()
        except:
            logging.debug("error loading data" + str(traceback.format_exc()))
            print("error loading data")
            return "error loading data"
        ManualQueryId = []
        user_id = req_data['user_id']
        user_data[str(user_id)] = req_data
        # print(req_data)
        ###########################################
        stop_process(temp_data, user_id)
        ###########################################
        temp_data = req_data
        for i in range(len(req_data['data'])):
            # print(req_data['data'])
            try:
                query = """Insert into manual_query (building_id,camera_id,start_time,end_time, x1, x2, y1, y2, h, w, is_processed, type) values ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s');""" % (
                    req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                    req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'], req_data['data'][i]['x1'], req_data['data'][i]['x2'],
                    req_data['data'][i]['y1'], req_data['data'][i]['y2'], req_data['data'][i]['h'],
                    req_data['data'][i]['w'], '0', 'box_motion_check')
                #print(query)
                # time.sleep(0.5)
                cur_COLOSQL.execute(query)
                db_conn_COLOSQL.commit()
                # logging.debug("insert successful into manual_query")
                #print("insert successful into manual_query")
            except:
                logging.debug(traceback.format_exc())
                db_conn_COLOSQL.rollback()
                print("error rollback")
            try:
                query = """select * from manual_query where building_id = '%s' and camera_id = '%s' and start_time = '%s' and end_time = '%s' and is_processed = 0;""" % (
                    req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                    req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'])
                #print(query)
                cur_COLOSQL.execute(query)
                manualquery_data = cur_COLOSQL.fetchall()
                for data in manualquery_data:
                    manualQueryID = data['id']
                    if manualQueryID not in ManualQueryId:
                        ManualQueryId.append(manualQueryID)
            except:
                logging.debug("Error in selecting from manual_query" + str(traceback.format_exc()))
                db_conn_COLOSQL.rollback()
            # executor.submit(send_manualQueryID, ManualQueryId)
        try:
            p = Process(target=send_manualQueryID, args=(req_data, user_id,))
            p.start()
            #print(p.pid)
            file_path = str(user_id) + ".txt"
            f = open(file_path, "w+")
            f.write(str(p.pid))
        except:
            logging.debug("Error in threading" + str(traceback.format_exc()))

        file_path = str(user_id) + ".txt"
        f = open(file_path, "w+")
        f.write(str(p.pid))
        #print(f.read())
        if req_data is not None:
            # logging.debug(jsonify(resp=True, ManualQueryId=ManualQueryId))
            return jsonify(resp=True, ManualQueryId=ManualQueryId)
        else:
            logging.debug(jsonify(resp=False, ManualQueryId=ManualQueryId))
            #print(False, ManualQueryId)
            return jsonify(resp=False, ManualQueryId=ManualQueryId)
    except:
        insert_into_exception("error in box motion api", str(traceback.format_exc()))


def send_manualQueryID(req_data, user_id):
    try:
        for i in range(len(req_data['data'])):
            try:
                (W, H) = (None, None)
                query = """select * from manual_query where building_id = '%s' and camera_id = '%s' and start_time = '%s' and end_time = '%s' and is_processed = 0;""" % (
                    req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                    req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'])
                cur_COLOSQL.execute(query)
                manualquery_data = cur_COLOSQL.fetchall()
                #print(query)
                query2 = """select * from videos where monitor_id = '%s' and start_time > '%s' and end_time < '%s'""" % (
                    req_data['data'][i]['camera_id'], req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'])
                cur_COLOSQL.execute(query2)
                rows_video = cur_COLOSQL.fetchall()
                #print(query2)
            except:
                logging.debug("error in loading data from table" + str(traceback.format_exc()))

            # not_video_availability = 1
                # print("error in loading data from the table")
            for data in manualquery_data:
                #print("hdjshafjhajkdhfjhaskdhfkj"+ str(data))
                manualQueryID = data['id']
                x1 = data['x1']
                y1 = data['y1']
                x2 = data['x2']
                y2 = data['y2']
                h = data['h']
                w = data['w']
                # dim = (w, h)
                min_area = 2500
                not_video_availability = 1
                for video in rows_video:
                    event_id = video["event_id"]
                    #print("Video name: ", video['video_name'], "\n\n\n")
                    # logging.debug("Video name" + video['video_name'])
                    datetime_object = datetime.strptime(video["start_time"], '%Y-%m-%d %H:%M:%S')
                    path = "/mnt/nas/zm/" + str(video['monitor_id']) + "/" + str(datetime_object.date()) + "/" + str(
                        video['event_id']) + "/" + str(video['video_name'])
                    if os.path.isfile(path):
                        if data['not_video_availability'] = 1 and not_video_availability = 1:
                        not_video_availability = 0
                        query_update = """UPDATE manual_query SET not_video_availability=0 WHERE id=%s""" % manualQueryID
                        cur_COLOSQL.execute(query_update)
                        db_conn_COLOSQL.commit()
                    #print(path)
                    vs = cv2.VideoCapture(path)
                    i = 0
                    motionFlag = False
                    initial_cropped_frame = None
                    while True:
                        (grabbed, frame) = vs.read()
                        i += 1
                        if not grabbed:
                            break
                        if i % 10 != 0:
                            #   print("ok")
                            continue
                        # print(w,h)
                        if w is 0 or h is 0:
                            (h, w) = frame.shape[:2]
                        if x2 is 0:
                            x2 = w
                        if y2 is 0:
                            y2 = h
                        width_ratio = frame.shape[1] / w
                        height_ratio = frame.shape[0] / h

                        new_x1 = round(x1 * width_ratio)
                        new_y1 = round(y1 * height_ratio)
                        new_x2 = round(x2 * width_ratio)
                        new_y2 = round(y2 * height_ratio)
                        #print(width_ratio, height_ratio, new_x1, new_x2, new_y1, new_y2)
                        cropped_frame = frame[new_y1:new_y2, new_x1:new_x2]

                        grayscale_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                        if initial_cropped_frame is None:
                            initial_cropped_frame = grayscale_cropped_frame
                            continue
                        cropped_frame_delta = cv2.absdiff(initial_cropped_frame, grayscale_cropped_frame)
                        thresholded_cropped_frame = cv2.threshold(cropped_frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
                        dilated_cropped_frame = cv2.dilate(thresholded_cropped_frame, None, iterations=2)
                        contours = cv2.findContours(dilated_cropped_frame.copy(), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                        contours = imutils.grab_contours(contours)

                        for contour in contours:
                            if cv2.contourArea(contour) < min_area:
                                continue
                            motionFlag = True
                            break
                        if motionFlag:
                            try:
                                query = """Insert into manual_query_result (manual_query_id, event_id) values ('%s', '%s');""" % (
                                    manualQueryID, event_id)
                                cur_COLOSQL.execute(query)
                                db_conn_COLOSQL.commit()
                                # logging.debug("inserted into manual_query_result")
                                #print(query)
                                break
                            except:
                                logging.debug(
                                    "Error in inserting into manual_query_result" + str(traceback.format_exc()))
                                print("error in inserting into manual_query_result")
                                db_conn_COLOSQL.rollback()
                        if motionFlag:
                            break
                try:
                    print("Updated manual_query")
                    queryUpdt = """UPDATE manual_query SET is_processed=1 WHERE id=%s""" % manualQueryID
                    cur_COLOSQL.execute(queryUpdt)
                    db_conn_COLOSQL.commit()
                except:
                    print(traceback.format_exc())
                    logging.debug("error in updating" + str(traceback.format_exc()))
                    db_conn_COLOSQL.rollback()
            return True
    except:
        insert_into_exception("error in subprocess box motion", str(traceback.format_exc()))


def extention_finder(encoded_string):
    decoded_string = b64decode(encoded_string)
    extention = imghdr.what(None, h=decoded_string)
    #print(extention)
    return extention


@app.route('/face_validate', methods=['POST'])
def prediction_upload():
    try:
        if request.method == 'POST':
            content = request.get_json()
            extention = extention_finder(content['img'])
            #print(extention)
            imgdata = base64.b64decode(content['img'])

            filename = "static/uploads." + str(extention)  # I assume you have a way of picking unique filenames
            with open(filename, 'wb') as f:
                f.write(imgdata)
            img_file = (os.path.abspath(filename))
            #print("ImgFile: ", img_file)

            di = {"data": faceValidate(filename)}
            response = app.response_class(
                response=json.dumps(di),
                status=200,
                mimetype='application/json'
            )
            #print(response)
            return response
    except:
        insert_into_exception("error in face_validation api", str(traceback.format_exc()))
        di = {"data": "Invalid Input"}
        response = app.response_class(
            response=json.dumps(di),
            status=422,
            mimetype='application/json'
        )
        return response


def faceValidate(imgPath):
    image = face_recognition.load_image_file(imgPath)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        return True
    else:
        return False
    #face_landmarks_list = face_recognition.face_landmarks(image)
    #li = []
    #showFlag = False
    #if len(face_locations) == 0:
    #    return False
    #if showFlag:
    #    for (i, rect) in enumerate(face_locations):
    #        cv2.rectangle(image, (rect[3], rect[0]), (rect[1], rect[2]), (0, 255, 0), 2)
    #for data in face_landmarks_list[0]:
    #    for (x, y) in face_landmarks_list[0][data]:
    #        li.append([x, y])
    #        if showFlag:
    #            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    #if showFlag:
    #    cv2.imshow("Output", image)
    #    cv2.waitKey(1500)

    #if len(li) == 72:
    #    return True
    #else:
    #    return False


@app.route('/manual_face_search', methods=['POST'])
def show_face_video():
    try:
        if request.method == 'POST':
            req_data = request.get_json()
            # print(req_data)
            #        exit()
            image = req_data["img"]
            extention = extention_finder(req_data['img'])
            #print(extention)
            user_id = req_data['user_id']
            user_data[str(user_id)] = req_data

            # print(image)
            # image = "b"+image
            # print(image)
            # image_path = os.path.join("static/image.jpg")
            # print(image_path)
            # exit()
            # imgdata = base64.b64decode(image)
            image_data = base64.b64decode(image)
            with open("static/image" + extention, 'wb') as fh:
                fh.write(image_data)
            frame = cv2.imread('static/image' + extention)
            # print(frame.shape)
            name = imgPred.pred_frame(frame)
            # if "unknown" not in name:
            image = fc.load_image_file("static/image"+extention)
            global p
            global temp_data
            try:
                req_data = request.get_json()
            except:
                logging.debug("error loading data" + str(traceback.format_exc()))
                print("error loading data")
                return "error loading data"
            ManualQueryId = []
            stop_process(temp_data, req_data['user_id'])
            temp_data = req_data
            for i in range(len(req_data['data'])):
                try:
                    query = """Insert into manual_query (building_id,camera_id,start_time,end_time, type) values ('%s', '%s', '%s', '%s', '%s');""" % (
                        req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                        req_data['data'][i]['start_time'],
                        req_data['data'][i]['end_time'], 'manual_face_search')
                    time.sleep(0.5)
                    cur_COLOSQL.execute(query)
                    db_conn_COLOSQL.commit()
                    # logging.debug("insert successful into manual_query")
                    print("insert successful into manual_query")
                except:
                    logging.debug(traceback.format_exc())
                    db_conn_COLOSQL.rollback()
                    print("error rollback")
                try:
                    query = """select * from manual_query where building_id = '%s' and camera_id = '%s' and start_time = '%s' and end_time = '%s' and is_processed = 0 and type = 'manual_face_search';""" % (
                        req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                        req_data['data'][i]['start_time'],
                        req_data['data'][i]['end_time'])
                    cur_COLOSQL.execute(query)
                    manualquery_data = cur_COLOSQL.fetchall()
                    for data in manualquery_data:
                        manualQueryID = data['id']
                        if manualQueryID not in ManualQueryId:
                            ManualQueryId.append(manualQueryID)
                except:
                    logging.debug("Error in selecting from manual_query" + str(traceback.format_exc()))
                    db_conn_COLOSQL.rollback()
            #print(name)
            if "unknown" not in name:
                print("will go in face_search")
                try:
                    p = Process(target=face_search, args=(req_data,))
                    p.start()
                except:
                    logging.debug("Error in threading of face search" + str(traceback.format_exc()))
            else:
                print("will go in face_match")
                try:
                    p = Process(target=face_match, args=(req_data, image,))
                    p.start()
                except:
                    logging.debug("Error in threading of face match" + str(traceback.format_exc()))

            file_path = str(user_id) + ".txt"
            f = open(file_path, "w+")
            f.write(str(p.pid))

            if req_data is not None:
                # logging.debug(jsonify(resp=True, ManualQueryId=ManualQueryId))
                return jsonify(resp=True, ManualQueryId=ManualQueryId)
            else:
                logging.debug(jsonify(resp=False, ManualQueryId=ManualQueryId))
                print(False, ManualQueryId)
                return jsonify(resp=False, ManualQueryId=ManualQueryId)
        # else:
    except:
        insert_into_exception("error in manual face api", str(traceback.format_exc()))


def face_search(req_data):
    try:
        for i in range(len(req_data['data'])):
            try:
                (W, H) = (None, None)
                query = """select * from manual_query where building_id = '%s' and camera_id = '%s' and start_time = '%s' and end_time = '%s' and is_processed = 0 and type = 'manual_face_search';""" % (
                    req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                    req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'])
                cur_COLOSQL.execute(query)
                manualquery_data1 = cur_COLOSQL.fetchall()
                query2 = """select * from videos where monitor_id = '%s' and start_time >= '%s' and end_time <= '%s' and is_motion = 1""" % (
                    req_data['data'][i]['camera_id'], req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'])
                cur_COLOSQL.execute(query2)
                rows_video = cur_COLOSQL.fetchall()
                #print(rows_video)
            except:
                logging.debug("error in loading data from table" + str(traceback.format_exc()))
            for data in manualquery_data1:
                manualQueryID = data['id']
                not_video_availability = 1
                for video in rows_video:

                    event_id = video["event_id"]
                    datetime_object = datetime.strptime(video["start_time"], '%Y-%I-%d %H:%M:%S')
                    startTime = datetime.timestamp(datetime_object)
                    path = "/mnt/nas/zm/" + str(video['monitor_id']) + "/" + str(datetime_object.date()) + "/" + str(
                        video['event_id']) + "/" + str(video['video_name'])
                    # path = "vlc-record-2019-11-21-10h27m10s-dshow___-.avi"
                    print(path)
                    if os.path.isfile(path):
                        if data['not_video_availability'] = 1 and not_video_availability = 1:
                        not_video_availability = 0
                        query_update = """UPDATE manual_query SET not_video_availability=0 WHERE id=%s""" % manualQueryID
                        cur_COLOSQL.execute(query_update)
                        db_conn_COLOSQL.commit()
                    vs = cv2.VideoCapture(path)  # data['video_name'])
                    i = 0
                    object_detection_datetime = 0
                    prevTime = 0
                    video_id = data['id']
                    while True:
                        li = []
                        dataLi = []
                        (grabbed, frame) = vs.read()
                        i += 1
                        if not grabbed:
                            break
                        if i % 10 != 0:
                            continue
                        name = imgPred.pred_frame(frame)
                        if name:
                            try:
                                query = """Insert into manual_query_result (manual_query_id, event_id) values ('%s', '%s');""" % (
                                    manualQueryID, event_id)
                                cur_COLOSQL.execute(query)
                                db_conn_COLOSQL.commit()
                                # logging.debug("inserted into manual_query_result")
                                break
                            except:
                                logging.debug(
                                    "Error in inserting into manual_query_result" + str(traceback.format_exc()))
                                print("error in inserting into manual_query_result")
                                db_conn_COLOSQL.rollback()
                try:
                    print("Updated manual_query")
                    queryUpdt = """UPDATE manual_query SET is_processed=1 WHERE id=%s""" % manualQueryID
                    cur_COLOSQL.execute(queryUpdt)
                    db_conn_COLOSQL.commit()
                except:
                    print(traceback.format_exc())
                    logging.debug("error in updating" + str(traceback.format_exc()))
                    db_conn_COLOSQL.rollback()
        return True
    except:
        insert_into_exception("error in face_search", str(traceback.format_exc()))


def face_match(req_data, input_frame):
    try:
        for i in range(len(req_data['data'])):
            try:
                (W, H) = (None, None)
                query = """select * from manual_query where building_id = '%s' and camera_id = '%s' and start_time = '%s' and end_time = '%s' and is_processed = 0 and type = 'manual_face_search';""" % (
                    req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                    req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'])
                print(query)
                cur_COLOSQL.execute(query)
                manualquery_data1 = cur_COLOSQL.fetchall()
                query2 = """select * from videos where monitor_id = '%s' and start_time >= '%s' and end_time <= '%s' and is_motion = 1""" % (
                    req_data['data'][i]['camera_id'], req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'])
                cur_COLOSQL.execute(query2)
                rows_video = cur_COLOSQL.fetchall()
            except:
                logging.debug("error in loading data from table" + str(traceback.format_exc()))
            for data in manualquery_data1:
                manualQueryID = data['id']
                not_video_availability = 1
                for video in rows_video:
                    event_id = video["event_id"]
                    datetime_object = datetime.strptime(video["start_time"], '%Y-%I-%d %H:%M:%S')
                    startTime = datetime.timestamp(datetime_object)
                    path = "/mnt/nas/zm/" + str(video['monitor_id']) + "/" + str(datetime_object.date()) + "/" + str(
                       video['event_id']) + "/" + str(video['video_name'])
                    #path = "vlc-record-2019-11-21-10h39m09s-window_frame-.mp4"
                    print(path)
                    if os.path.isfile(path):
                        if data['not_video_availability'] = 1 and not_video_availability = 1:
                        not_video_availability = 0
                        query_update = """UPDATE manual_query SET not_video_availability=0 WHERE id=%s""" % manualQueryID
                        cur_COLOSQL.execute(query_update)
                        db_conn_COLOSQL.commit()
                    vs = cv2.VideoCapture(path)  # data['video_name'])
                    i = 0
                    while True:
                        (grabbed, frame) = vs.read()
                        i += 1
                        if not grabbed:
                            break
                        if i % 10 != 0:
                            continue
                        new_face = input_frame
                        check_face = frame
                        #print(new_face.shape)
                        #print(check_face)
                        #print("this is check" + str(face_recognition.face_encodings(new_face)))
                        try:
                            image = face_recognition.load_image_file(new_face)
                            biden_encoding = face_recognition.face_encodings(image)[0]
                            unknown_encoding = face_recognition.face_encodings(check_face)[0]
                        except:
                            print(traceback.format_exc())
                            break
                        # print(known_image)
                        results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
                        #print("result" + str(results))
                        if results:
                            try:
                                query = """Insert into manual_query_result (manual_query_id, event_id) values ('%s', '%s');""" % (
                                    manualQueryID, event_id)
                                cur_COLOSQL.execute(query)
                                db_conn_COLOSQL.commit()
                                print(query)
                                # logging.debug("inserted into manual_query_result")
                                break
                            except:
                                logging.debug(
                                    "Error in inserting into manual_query_result" + str(traceback.format_exc()))
                                print("error in inserting into manual_query_result")
                                db_conn_COLOSQL.rollback()
                try:
                    print("Updated manual_query")
                    queryUpdt = """UPDATE manual_query SET is_processed=1 WHERE id=%s""" % manualQueryID
                    cur_COLOSQL.execute(queryUpdt)
                    db_conn_COLOSQL.commit()
                except:
                    print(traceback.format_exc())
                    logging.debug("error in updating" + str(traceback.format_exc()))
                    db_conn_COLOSQL.rollback()
        return True
    except:
        insert_into_exception("error in face_match", str(traceback.format_exc()))


@app.route("/lpr", methods=['POST'])
def receive_and_response_lpr():
    try:
        global p
        global temp_data
        try:
            req_data = request.get_json()
        except:
            logging.debug("error loading data" + str(traceback.format_exc()))
            print("error loading data")
            return "error loading data"
        ManualQueryId = []
        user_id = req_data['user_id']
        user_data[str(user_id)] = req_data
        stop_process(temp_data, req_data['user_id'])
        temp_data = req_data
        for i in range(len(req_data['data'])):
            try:
                query = """Insert into manual_query (building_id,camera_id,start_time,end_time, x1, x2, y1, y2, h, w, is_processed, type) values ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s');""" % (
                    req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                    req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'], req_data['data'][i]['x1'], req_data['data'][i]['x2'],
                    req_data['data'][i]['y1'], req_data['data'][i]['y2'], req_data['data'][i]['h'],
                    req_data['data'][i]['w'], '0', 'lpr')
                time.sleep(0.5)
                cur_COLOSQL.execute(query)
                db_conn_COLOSQL.commit()
                # logging.debug("insert successful into manual_query")
                print("insert successful into manual_query")
            except:
                logging.debug(traceback.format_exc())
                db_conn_COLOSQL.rollback()
                print("error rollback")
            try:
                query = """select * from manual_query where building_id = '%s' and camera_id = '%s' and start_time = '%s' and end_time = '%s' and is_processed = 0 and type = 'lpr';""" % (
                    req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                    req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'])
                cur_COLOSQL.execute(query)
                manualquery_data = cur_COLOSQL.fetchall()
                for data in manualquery_data:
                    manualQueryID = data['id']
                    if manualQueryID not in ManualQueryId:
                        ManualQueryId.append(manualQueryID)
            except:
                logging.debug("Error in selecting from manual_query" + str(traceback.format_exc()))
                db_conn_COLOSQL.rollback()
        try:
            p = Process(target=lpr_apply, args=(req_data,))
            p.start()
        except:
            logging.debug("Error in threading" + str(traceback.format_exc()))
        user_id = req_data['user_id']
        #print(p.pid)
        # print(os.getpid())
        file_path = str(user_id) + ".txt"
        f = open(file_path, "w+")
        f.write(str(p.pid))
        if req_data is not None:
            # logging.debug(jsonify(resp=True, ManualQueryId=ManualQueryId))
            return jsonify(resp=True, ManualQueryId=ManualQueryId)
        else:
            logging.debug(jsonify(resp=False, ManualQueryId=ManualQueryId))
            print(False, ManualQueryId)
            return jsonify(resp=False, ManualQueryId=ManualQueryId)
    except:
        insert_into_exception("error in LPR API", str(traceback.format_exc()))


def lpr_apply(req_data):
    try:
        # print("in child")
        # print(os.getpid)
        global number_plate
        for i in range(len(req_data['data'])):
            try:
                query = """select * from manual_query where building_id = '%s' and camera_id = '%s' and start_time = '%s' and end_time = '%s' and is_processed = 0 and type = 'lpr';""" % (
                    req_data['data'][i]['building_id'], req_data['data'][i]['camera_id'],
                    req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'])
                cur_COLOSQL.execute(query)
                manualquery_data = cur_COLOSQL.fetchall()
                query2 = """select * from videos where monitor_id = '%s' and start_time >= '%s' and end_time <= '%s' and is_motion = 1""" % (
                    req_data['data'][i]['camera_id'], req_data['data'][i]['start_time'],
                    req_data['data'][i]['end_time'])
                cur_COLOSQL.execute(query2)
                rows_video = cur_COLOSQL.fetchall()
            except:
                logging.debug("error in loading data from table" + str(traceback.format_exc()))
            for data in manualquery_data:
                manualQueryID = data['id']
                x1 = data['x1']
                y1 = data['y1']
                x2 = data['x2']
                y2 = data['y2']
                h = data['h']
                w = data['w']
                manual_query_id_lpr = None
                not_video_availability = 1
                for video in rows_video:
                    event_id = video["event_id"]
                    print("Video name: ", video['video_name'], "\n\n\n")
                    # logging.debug("Video name" + video['video_name'])
                    datetime_object = datetime.strptime(video["start_time"], '%Y-%m-%d %H:%M:%S')
                    startTime = datetime.timestamp(datetime_object)
                    path = "/mnt/nas/zm/" + str(video['monitor_id']) + "/" + str(datetime_object.date()) + "/" + str(
                        video['event_id']) + "/" + str(video['video_name'])
                    # path = "videoplayback.mp4"
                    print(path)
                    if os.path.isfile(path):
                        if data['not_video_availability'] = 1 and not_video_availability = 1:
                        not_video_availability = 0
                        query_update = """UPDATE manual_query SET not_video_availability=0 WHERE id=%s""" % manualQueryID
                        cur_COLOSQL.execute(query_update)
                        db_conn_COLOSQL.commit()
                    vs = cv2.VideoCapture(path)
                    anything_detection = 0
                    j = 0
                    li = []
                    objLi = []
                    while True:
                        (grabbed, frame) = vs.read()
                        # frame_new = frame
                        #frame = cv2.imread("889212bc2b99ab979b9ecc3e5b6a8a4dx.jpg")
                        j += 1
                        if not grabbed:
                            break
                        if j % 10 != 0:
                            continue
                        if w is 0 or h is 0:
                            (h, w) = frame.shape[:2]
                        if x2 is 0:
                            x2 = w
                        if y2 is 0:
                            y2 = h
                        width_ratio = frame.shape[1] / w
                        height_ratio = frame.shape[0] / h
                        new_x1 = round(x1 * width_ratio)
                        new_y1 = round(y1 * height_ratio)
                        new_x2 = round(x2 * width_ratio)
                        new_y2 = round(y2 * height_ratio)
                        #print(width_ratio, height_ratio, new_x1, new_x2, new_y1, new_y2)
                        cropped_frame = frame[new_y1:new_y2, new_x1:new_x2]
                        blob = cv2.dnn.blobFromImage(cropped_frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
                        net.setInput(blob)
                        layerOutputs = net.forward(ln)
                        boxes = []
                        confidences = []
                        classIDs = []
                        for output in layerOutputs:
                            for detection in output:
                                scores = detection[5:]
                                classID = np.argmax(scores)
                                confidence = scores[classID]

                                if confidence > 0.5:
                                    box = detection[0:4] * np.array([w, h, w, h])
                                    (centerX, centerY, width, height) = box.astype("int")
                                    x = int(centerX - (width / 2))
                                    y = int(centerY - (height / 2))
                                    boxes.append([x, y, int(width), int(height)])
                                    confidences.append(float(confidence))
                                    classIDs.append(classID)
                        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
                        if len(idxs) > 0:
                            for k in idxs.flatten():
                                obj = LABELS[classIDs[k]]
                                #print(obj)
                                if obj in ['car', 'bus', 'truck', 'motorbike']:
                                    try:
                                        y_1 = boxes[k][0]
                                        x_1 = boxes[k][1]
                                        w_1 = boxes[k][2]
                                        h_1 = boxes[k][3]
                                        #print(y_1, x_1, w_1, h_1)
                                        frame_crop = cropped_frame[y_1:y_1 + h_1, x_1:x_1 + w_1]
                                        object_detection_datetime_video = datetime.fromtimestamp(
                                            startTime + vs.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                                        ret, enc = cv2.imencode("*.jpg", cropped_frame)
                                        results = alpr.recognize_array(bytes(bytearray(enc)))
                                        print(results)
                                    except:
                                        continue
                                    l = 0
                                    for plate in results['results']:
                                        l += 1
                                        for candidate in plate['candidates']:
                                            prefix = "-"
                                            if candidate['matches_template']:
                                                prefix = "*"
                                            number_plate = candidate['plate']
                                            # else:
                                            #    number_plate =
                                    print("plate" + str(number_plate))
                                    if number_plate not in li and number_plate not in objLi:
                                        objLi = [event_id, obj, object_detection_datetime_video, number_plate]
                                        li.append(objLi)
                                        print(li)
                                        anything_detection += 1
                            # print("ok")
                    if anything_detection > 1:
                        try:
                            query = """Insert into manual_query_result (manual_query_id, event_id) values ('%s', '%s');""" % (
                                manualQueryID, event_id)
                            cur_COLOSQL.execute(query)
                            db_conn_COLOSQL.commit()
                            print(query)
                        except:
                            logging.debug(
                                "Error in inserting into manual_query_result" + str(traceback.format_exc()))
                            print("error in inserting into manual_query_result")
                            db_conn_COLOSQL.rollback()
                    try:
                        query = """select * from manual_query_result where manual_query_id = '%s' and event_id = '%s' order by id DESC limit 1;""" % (
                            manualQueryID, event_id)
                        cur_COLOSQL.execute(query)
                        manual_query_result = cur_COLOSQL.fetchall()
                        for item in manual_query_result:
                            manual_query_id_lpr= item['id']
                        print(manual_query_id_lpr)
                    except:
                        logging.debug("Error in fetching data" + str(traceback.format_exc()))
                        db_conn_COLOSQL.rollback()
                        print("error rollback")
                    try:
                        for items in li:
                            query2 = """Insert into lpr (id_mqr, detection_datetime, object_name, license_number) values ('%s', '%s', '%s', '%s');""" % (
                                manual_query_id_lpr, items[2], items[1], items[3])
                            cur_COLOSQL.execute(query2)
                            db_conn_COLOSQL.commit()
                            # logging.debug("inserted into LPR")
                    except:
                        logging.debug("Error in inserting into LPR" + str(traceback.format_exc()))
                        print("error in inserting into LPR")
                        db_conn_COLOSQL.rollback()
            try:
                print("Updated manual_query")
                queryUpdt = """UPDATE manual_query SET is_processed=1 WHERE id=%s""" % manualQueryID
                cur_COLOSQL.execute(queryUpdt)
                db_conn_COLOSQL.commit()
            except:
                print(traceback.format_exc())
                logging.debug("error in updating" + str(traceback.format_exc()))
                db_conn_COLOSQL.rollback()
        return True
    except:
        insert_into_exception("error in LPR subprocess", str(traceback.format_exc()))


def video_to_frames(video_filename, name):
    """Extract frames from video"""
    # video_filename = "https://muszm.s3-us-west-2.amazonaws.com/157/2/2020-01-08/501098-video.mp4"
    cap = cv2.VideoCapture(video_filename)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    i = 0
    if cap.isOpened() and video_length > 0:
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (300, 210))
            if ret:
                cv2.imwrite('static/%s.jpg' % name, frame)
                cap.release()
                break
            else:
                break
        return "/static/" + name + ".jpg"
    else:
        return False


@app.route('/frame_extractor', methods=['GET', 'POST'])
def form_example():
    try:
        folder = 'static'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        req_data = request.get_json()
        my_dictionary = {}
        my_list = []
        for i in range(len(req_data['urls'])):
            my_dict = {}
            camera_name = req_data['urls'][i]['cameraName']
            print(camera_name)
            camera_name = camera_name.replace(" ", "_")
            print(camera_name)
            image_url = video_to_frames(req_data['urls'][i]["url"], camera_name)
            my_dict["cameraId"] = req_data['urls'][i]['cameraId']
            my_dict["cameraName"] = req_data['urls'][i]['cameraName']
            print(req_data)
            if not image_url:
                continue
                # my_dict["url"] = "Invalid Video Entry"

                # continue
            else:
                # my_dict["cameraId"] = req_data['urls'][i]['cameraId']
                # my_dict["cameraName"] = req_data['urls'][i]['cameraName']
                my_dict["url"] = image_url
            print(my_dict)
            my_list.append(my_dict)
            my_dictionary['results'] = my_list
            app_json = jsonify(my_dictionary)
        return app_json
    except:
        insert_into_exception("error in frame_extractor", str(traceback.format_exc()))
        return "error in frame_extractor" + str(traceback.format_exc())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
