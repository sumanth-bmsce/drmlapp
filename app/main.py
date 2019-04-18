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

        # For the 1st image
        file1 = request.files["img1"]
        filename1 = secure_filename(file1.filename)
        path1 = os.path.join(app.config['UPLOAD_FOLDER'],filename1)
        file1.save(path1)
        print(path1)

        # For the 2nd image
        file2 = request.files["img2"]
        filename2 = secure_filename(file2.filename)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'],filename2)
        file2.save(path2)
        print(path2)

        #Loading 1st image
        y1 = image.load_img(path1, target_size=(299, 299, 3))
        y1 = image.img_to_array(y1)
        y1 = np.expand_dims(y1, axis=0)
        y1 = preprocess_input(y1)

        #Loading 2nd image
        y2 = image.load_img(path2, target_size=(299, 299, 3))
        y2 = image.img_to_array(y2)
        y2 = np.expand_dims(y2, axis=0)
        y2 = preprocess_input(y2)
        #print(x2)

        array1 = np.zeros(shape=(2, 299, 299, 3))
        array1[0] = y1
        array1[1] = y2
        print("Loading Complete Inception")

        #Loading 1st image
        x1 = image.load_img(path1, target_size=(224, 224, 3))
        x1 = image.img_to_array(x1)
        x1 = np.expand_dims(x1, axis=0)
        x1 = preprocess_input(x1)

        #Loading 2nd image
        x2 = image.load_img(path2, target_size=(224, 224, 3))
        x2 = image.img_to_array(x2)
        x2 = np.expand_dims(x2, axis=0)
        x2 = preprocess_input(x2)
        #print(x2)

        array = np.zeros(shape=(2, 224, 224, 3))
        array[0] = x1
        array[1] = x2
        print("Loading Complete Vgg")

        # Obtaining VGG16 Convolutions
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

        #Obtaining Inception Convolutions
        print("Constructing Inception v3 Model .....")
        model_inception = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="avg")
        flatten_model_inception = Sequential()
        flatten_model_inception.add(model_inception)
        print("Predicting Inception v3")
        features_input_inception = flatten_model_inception.predict(array1, verbose=1, batch_size=4)
        print("Extracted Inception v3 features shape ", features_input_inception.shape)
        print("Inception v3 Convolutions Done and Returned")
        print(features_input_inception)

        #Loading VGG16 Model
        print("Loading VGG16 model")
        new_model_vgg = load_model("oversample_model_train005_0_1427_512_100_epochs.h5")
        y_pred_vgg = new_model_vgg.predict_proba(features_input_vgg)
        #y_pred_inception = new_model_inception.predict_proba(features_inception)
        print("Probability list returned to predict")
        print(y_pred_vgg)

        #Loading Inception Model
        new_model_inception = load_model("inceptionv3_train.h5")
        y_pred_inception = new_model_inception.predict_proba(features_input_inception)
        print("Probability list returned to predict")
        print(y_pred_inception)

        predict_list = []
        print("Predicting Class .....")
        for vgg_list, inception_list in zip(y_pred_vgg, y_pred_inception):
            vgg_list = np.ndarray.tolist(vgg_list)
            inception_list = np.ndarray.tolist(inception_list)
            if (vgg_list.index(max(vgg_list)) == 0):
                predict_list.append(vgg_list.index(max(vgg_list)))
            else:
                if (vgg_list.index(max(vgg_list)) == inception_list.index(max(inception_list))):
                    predict_list.append(vgg_list.index(max(vgg_list)))
                else:
                    if (vgg_list[vgg_list.index(max(vgg_list))] > inception_list[vgg_list.index(max(inception_list))]):
                        predict_list.append(vgg_list.index(max(vgg_list)))
                    else:
                        predict_list.append(inception_list.index(max(inception_list)))
        print("Class Prediction Complete")
        print("Predicted list")
        print(predict_list)

        return render_template("display.html", variable = predict_list)

if __name__ == '__main__':
    app.run(host="10.130.29.98",port=8080,debug=True)

