from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

#Tensorflow 
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)


model = load_model('ResNet50Segmentation.h5')
resnetNSmodel = load_model('ResNet50NS.h5')
print('Model loaded. Check http://127.0.0.1:5000/ or http://localhost:5000/')


def predict_model(file_path, model):
    image_chosen = image.load_img(file_path, target_size=(224, 224))


    # Preprocessing the image
    image_x = image.img_to_array(image_chosen)
    # x = np.true_divide(x, 255)
    image_x = np.expand_dims(image_x, axis=0)

    image_x = tf.keras.applications.resnet.preprocess_input(image_x, data_format=None)

    preds = model.predict(image_x)
    return preds

def predict_model2(file_path, model):
    image_chosen = image.load_img(file_path, target_size=(224, 224))
    # Preprocessing the image
    image_x = image.img_to_array(image_chosen)
    # x = np.true_divide(x, 255)
    image_x = np.expand_dims(image_x, axis=0)
    
    preds = resnetNSmodel.predict(image_x)

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('classifierexS.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file_x = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file_x.filename))
        file_x.save(file_path)
        # Make prediction
        preds = predict_model(file_path, model)

        preds = tf.nn.sigmoid(preds)
        #pred = tf.where(pred < 0.5, 0, 1)
        print(preds)
        print("this is model1")
        
        if preds[0][0] >0.5:
            result = 'This image is correct to calculate an accurate CRL.'
        else:
            result ='This image is incorrect to calculate an accurate CRL. Please take another scan.'

        return result
    return None




@app.route('/NonSeg', methods=['GET'])
def indexNS():
    # Main page
    return render_template('classifierexNS.html')

@app.route('/predictNonSeg', methods=['GET', 'POST'])
def uploadNS():
    if request.method == 'POST':
        # Get the file from post request
        file_x = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file_x.filename))
        file_x.save(file_path)
        # Make prediction
        preds = predict_model2(file_path, model)

        preds = tf.nn.sigmoid(preds)
        #pred = tf.where(pred < 0.5, 0, 1)
        print(preds)
        
        
        if preds[0][0] >0.5:
            result = 'This image is correct to calculate an accurate CRL.'
        else:
            result ='This image is incorrect to calculate an accurate CRL. Please take another scan.'

        return result
    return None

if __name__ == '__main__':
    app.run(port=5002, debug=True)
    #Serve the app with gevent
    #http_server = WSGIServer(('localhost', 5000), app)
    #http_server.serve_forever()
    app.run()
