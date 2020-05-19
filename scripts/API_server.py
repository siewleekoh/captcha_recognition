import cherrypy
import os
from paste.translogger import TransLogger
from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
from helper_function import img_process


# working directory
captcha_home_path = os.path.abspath(__file__ + "/../../")
model_label_path = captcha_home_path + '/model/model_labels.dat'
model_path = captcha_home_path + '/model/captcha_model.hdf5'


# function to predict 4 letters
def captcha_solver(img_path, size_x, size_y, model_path, model_label_path):
    # Load up the model labels (so we can translate model predictions to actual letters)
    with open(model_label_path, "rb") as f:
        lb = pickle.load(f)
    model = tf.keras.models.load_model(model_path)

    predictions = []

    # Load the image and convert it to grayscale and then numpy array
    image = Image.open(img_path).convert("L")  # Grayscale conversion

    cropped_image1 = np.array(image.crop((10, 0, 50, 70)))
    cropped_image2 = np.array(image.crop((45, 0, 85, 70)))
    cropped_image3 = np.array(image.crop((80, 0, 120, 70)))
    cropped_image4 = np.array(image.crop((120, 0, 160, 70)))
    cropped_image = [cropped_image1, cropped_image2, cropped_image3, cropped_image4]

    # Save out each letter as a single image
    for ind_img in cropped_image:
        image = img_process(ind_img, size_x, size_y)

        # Add a third channel dimension to the image to make Keras happy
        image = np.expand_dims(image, axis=2)
        image = np.expand_dims(image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(image)
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    return captcha_text


def create_app():
    # app initialization
    app = Flask(__name__)
    # Cross Domain
    CORS(app)


    @app.route('/', methods=['GET'])
    def index():
        """
        hello world testing endpoint
        """
        return 'Hello World!'


    @app.route('/captcha/predict',  methods=['POST'])
    def predict_captcha():
        try:
            req_data = request.json
            captcha_url = req_data['captchafile']
            # download image and save it
            urllib.request.urlretrieve(captcha_url, captcha_home_path + '/model/temp_captcha.jpg')
            img_path = captcha_home_path + '/model/temp_captcha.jpg'
            captcha_text = captcha_solver(img_path, 28, 28, model_path, model_label_path)

        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            # Display an error to the console
            captcha_text = 'An error has occurred: ' + str(e)  # send to API
            print('An error has occurred.', str(e))  # print to console

        return jsonify(captcha_text)

    return app


def run_server(app):
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)

    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')

    CHERRYPY_config = {
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 8080,
        'server.socket_host': 'localhost'
    }

    # Set the configuration of the web server
    cherrypy.config.update(CHERRYPY_config)
    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == "__main__":
    # start flask
    app = create_app()

    # start web server
    run_server(app)
