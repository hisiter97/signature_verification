import os
import urllib.request

import cv2
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from predict import SigVerfiy
from flask import Flask

sig_verifier = SigVerfiy(weight_path='models/signet_f_lambda_0.95.pth')
UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1006 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["CACHE_TYPE"] = "null"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file_1' not in request.files:
        flash('No file part')
        return redirect(request.url)
    if 'file_2' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file_1 = request.files['file_1']
    file_2 = request.files['file_2']
    file_names = []
    if file_1:
        if file_1 and allowed_file(file_1.filename):
            # filename = secure_filename(file_1.filename)
            filename = 'sig1.jpg'
            file_names.append(filename)
            file_1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    if file_2:
        if file_2 and allowed_file(file_2.filename):
            # filename = secure_filename(file_2.filename)
            filename = 'sig2.jpg'
            file_names.append(filename)
            file_2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    if len(file_names) == 2:
        sig1 = os.path.join(app.config['UPLOAD_FOLDER'],file_names[0])
        sig2 = os.path.join(app.config['UPLOAD_FOLDER'],file_names[1])
    else:
        return redirect(request.url)

    result, label, confidence = sig_verifier.predict(sig1, sig2, visualize=False)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'res.jpg')
    cv2.imwrite(result_path, result)

    # else:
    #	flash('Allowed image types are -> png, jpg, jpeg, gif')
    #	return redirect(request.url)
    print(confidence)
    return render_template('upload.html', filenames=file_names, label=label, result_path='res.jpg', confidence=str(confidence),
                           threshold=sig_verifier.threshold)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(host='localhost', port=8889, debug=True)