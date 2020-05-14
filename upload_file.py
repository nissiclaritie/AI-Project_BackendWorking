from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import face_validation.face_validation as fd
from face_validation import skin_tone as st
from face_validation import whitebalance_fix as wt
import cv2 as cv

app = Flask(__name__)

uploads_dir = 'static'
print(uploads_dir)
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)


@app.route('/upload')
def upload_file():
    return render_template("index.html")


@app.route('/uploader', methods=['GET', 'POST'])
def post_file_upload():
    if request.method == "POST":
        f = request.files['file']
        f.save(os.path.join(uploads_dir, secure_filename(f.filename)))
        file_name = os.path.join(uploads_dir, secure_filename(f.filename))
        angle, shape, image = fd.process_file(file_name)
        image = wt.white_balance(image)
        cv.imwrite(str(file_name), image)
        try:
            rois_head, roi_left_cheek, roi_right_cheek = fd.roi_face(angle, shape, image)
        except:
            rois_head = roi_left_cheek = roi_right_cheek = 1
        tone = 'None'
        try:
            if rois_head != None and roi_left_cheek != None and roi_right_cheek != None:
                # st.image_rgb(rois_head, roi_left_cheek, roi_right_cheek)
                tone = 'None'
        except Exception as ex:
            tone = st.image_rgb(rois_head, roi_left_cheek, roi_right_cheek)
        angle['Tone'] = tone
        return angle


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8010, debug=True)
