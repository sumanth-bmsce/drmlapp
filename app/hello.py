from werkzeug import secure_filename
from flask import Flask, render_template, request
import numpy as np
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def VGGConv(image_array):
    print("Loading Images .....")
    array = np.zeros(shape=(1, 224, 224, 3))
    array[0] = image_array
    print("Loading Complete")
    print("Constructing VGG16 Model .....")
    model_vgg = VGG16(include_top = False, weights = 'imagenet',input_shape = (224,224,3))
    flatten_model_vgg = Sequential()
    flatten_model_vgg.add(model_vgg)
    flatten_model_vgg.add(Flatten())
    print("Predicting VGG16")
    features_input_vgg = flatten_model_vgg.predict(array, verbose = 1,batch_size = 4)
    print("Extracted VGG16 features shape ",features_input_vgg.shape)
    print("VGG16 Convolutions Done and Returned")
    print("VGG Features")
    print(features_input_vgg)

@app.route("/")
def homepage():
    return render_template("homepage.html")

@app.route("/result", methods = ['POST', 'GET'])
def result():
    if(request.method=="POST"):
        result = request.form
        return render_template("result.html", result=result)

@app.route("/display", methods = ['POST', 'GET'])

def navigate():
    if(request.method=="POST"):
        file = request.files["img"]
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        file.save(path)
        print(path)
        x = image.load_img(path, target_size=(224, 224, 3))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        print(x)
        array = np.zeros(shape=(1, 224, 224, 3))
        array[0] = x
        print("Loading Complete")
        print("Constructing VGG16 Model .....")
        model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        flatten_model_vgg = Sequential()
        flatten_model_vgg.add(model_vgg)
        flatten_model_vgg.add(Flatten())
        print("Predicting VGG16")
        features_input_vgg = flatten_model_vgg.predict(array, verbose=1, batch_size=1)
        print("Extracted VGG16 features shape ", features_input_vgg.shape)
        print("VGG16 Convolutions Done and Returned")
        print("VGG Features")
        print(features_input_vgg)
        print("Loading VGG16 model")
        new_model_vgg = load_model("oversample_model_train005_0_1427_512.h5")
        y_pred_vgg = new_model_vgg.predict_proba(features_input_vgg)
        #y_pred_inception = new_model_inception.predict_proba(features_inception)
        print("Probability list returned to predict")
        print(y_pred_vgg)
        prediction = y_pred_vgg.argmax(axis=1)
        print(prediction)
        pred = prediction[0]
        print(pred)
        return render_template("display.html", variable = pred)

if __name__ == '__main__':
    app.run(debug=True)

