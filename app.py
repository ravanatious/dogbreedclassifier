from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template
from tensorflow import keras
from keras.applications.inception_v3 import decode_predictions, preprocess_input, InceptionV3
import cv2
import csv
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'kekw_lol_haha hehe lolololo nopers 23124215134213'
bootstrap = Bootstrap(app)
model = keras.models.load_model('static/inceptionv3_rev2')
filepath = 'dogclassification_breed.csv'


#image upload form which checks for empty or non-image file types
class UploadForm(FlaskForm):
    upload = FileField('Select an image:', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png', 'jpeg', 'JPEG', 'PNG', 'JPG'], 'Images only!')
    ])
    submit = SubmitField('Classify')

#read csv to get list of classes used for training the model
def read_class_csv(filename):
    file = open(filename)
    csvreader = csv.reader(file)

    csvlist = []
    for row in csvreader:
        csvlist.append(row)
    file.close()
    return csvlist


#predict the image 
def get_prediction(img_path, breeds):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (299, 299))
    img_preprocessed = preprocess_input(img_resized)
    img_reshaped = img_preprocessed.reshape((1, 299, 299, 3))
    prediction = model.predict(img_reshaped)
    breed = breeds[np.argmax(prediction)] 
    pred_score = round(prediction[0][np.argmax(prediction)] * 100,2)

    
    return breed, pred_score


@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    
    if form.validate_on_submit():
        f = form.upload.data
        filename = secure_filename(f.filename)
        file_url = os.path.join('static', filename
        )
        f.save(file_url)
        form = None
        breedlist = read_class_csv(filepath)
        breed, pred_score = get_prediction(file_url,breedlist)
    else:
        file_url = None
        breed = None
        pred_score = None
    return render_template("home.html", form=form, file_url=file_url, breed=breed, pred_score=pred_score)


if __name__ == "__main__":
    app.run(debug=True)