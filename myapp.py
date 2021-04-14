import os
from flask import Flask, render_template, request
from flask import send_from_directory
import tensorflow
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
MODEL_FOLDER = 'static/model'

# Load And Prepare The Image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(200, 200))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 200, 200, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

# load an image and predict the class
def predict(filename):
    # load the image
    img = load_img(filename, target_size=(200, 200))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 200, 200, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    # load model
    model = load_model(MODEL_FOLDER + '/model_catsVSdogs.h5')
    # predict the class
    result = model.predict(img)
    print(result)
    result = result.flatten()
    result = round(result[0])
    K.clear_session()
    return result


# home page
@app.route('/')
def home():
   return render_template('welcome.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('upload.html')
    else:
        file = request.files['filename']
        file.save(UPLOAD_FOLDER + file.filename)
        # full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        # file.save(full_name)

        indices = {0: 'Cat', 1: 'Dog'}
        result = predict(UPLOAD_FOLDER + file.filename)

        #accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[result]

    return render_template('upload.html', filename = file.filename, label = result)


# @app.route('/uploads/<filename>')
# def send_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

