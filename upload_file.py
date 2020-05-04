from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import face_validation.face_validation as fd
from face_validation import  skin_tone as st


app = Flask(__name__)

# uploads_dir = os.path.join(app.instance_path, 'uploads')
uploads_dir = 'static'
print(uploads_dir)
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
# exit()


@app.route('/upload')
def upload_file():
    return render_template("index.html")


@app.route('/uploader', methods=['GET', 'POST'])
def post_file_upload():
    if request.method == "POST":
        f = request.files['file']
        f.save(os.path.join(uploads_dir, secure_filename(f.filename)))
        # exit()
        angle, shape,image = fd.process_file(os.path.join(uploads_dir, secure_filename(f.filename)))
        rois = fd.roi_face(angle,shape,image)
        tone = 'None'
        try:
            if rois !=None:
                st.image_rgb(rois)
                tone = 'None'
        except Exception as ex:
            tone = st.image_rgb(rois)
            print(tone)
            print(ex)
        angle['Tone'] = tone
        return angle


if __name__ == '__main__':
    app.run(port=5500, debug=True)
