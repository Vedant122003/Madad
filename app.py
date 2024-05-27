from flask import Flask, request, render_template, redirect, session, url_for, flash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import firebase_admin
from firebase_admin import credentials, storage
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from werkzeug.utils import secure_filename
import os
import pyrebase
from flask import Flask, flash, session, redirect, url_for, request, render_template
from flask import Flask, request, jsonify


from flask import Flask, request, jsonify, render_template
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from sklearn.metrics import accuracy_score




app = Flask(__name__)
app.secret_key = 'your-very-secret-key'

# Initialize Firebase Admin SDK
firebase_sdk_path = r''
cred = credentials.Certificate(firebase_sdk_path)
firebase_app = firebase_admin.initialize_app(cred, {
    'storageBucket': ''
})

config = {
    "apiKey": "",
    "authDomain": "",
    "databaseURL": "",
    "projectId": "",
    "storageBucket": "",
    "messagingSenderId": "",
    "appId": "7",
    "measurementId": ""
}

firebase = pyrebase.initialize_app(config)
firebase_auth = firebase.auth()
firebase_db = firebase.database()
bucket = storage.bucket()   
firebase = pyrebase.initialize_app(config)
firebase_auth = firebase.auth()


# Assuming the model path and class names are set
model_path = r""
model = load_model(model_path) if os.path.exists(model_path) else None
class_names = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']



@app.route('/', methods=['GET'])
def welcome():
    return render_template('welcome.html')

@app.route('/login_signup', methods=['GET', 'POST'])
def login_signup():
    if request.method == 'POST':
        action = request.form['action']
        email = request.form['email']
        password = request.form['password']

        if action == 'signup':
            try:
                user = firebase_auth.create_user_with_email_and_password(email, password)
                flash('User successfully registered. Please log in.')
                return redirect(url_for('login_signup'))
            except Exception as e:
                flash('Failed to register. Please check your details and try again.')
        
        elif action == 'signin':
            try:
                user = firebase_auth.sign_in_with_email_and_password(email, password)
                session['user'] = user['idToken']
                return redirect(url_for('homepage'))  # Redirect to homepage.html
            except Exception as e:
                flash('Invalid email or password.')

    return render_template('login.html')

@app.route('/signout')
def signout():
    session.pop('user', None)  # Clear the user session
    flash('You have been logged out.')
    return redirect(url_for('login_signup'))


@app.route('/homepage')
def homepage():
    if 'user' in session:
        return render_template('homepage.html')
    else:
        flash("Please log in to view this page.")
        return redirect(url_for('login_signup'))



@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files and 'imageData' not in request.form:
        return 'No image provided', 400

    file = request.files.get('file')
    if file and file.filename:
        filename = secure_filename(file.filename)
        blob = bucket.blob(filename)
        blob.upload_from_string(file.read(), content_type=file.content_type)
        image_url = blob.public_url
    elif 'imageData' in request.form:
        encoded_data = request.form['imageData'].split(',')[1]
        image_data = base64.b64decode(encoded_data)
        filename = 'captured_image.png'
        blob = bucket.blob(filename)
        blob.upload_from_string(image_data, content_type='image/png')
        image_url = blob.public_url
    else:
        return 'No valid image found', 400

    image_path = download_blob_to_tmp(blob)
    predicted_soil_type, confidence = predict_soil_type(image_path)
    if confidence < 0.99:
        return render_template('index.html', error_message='low_confidence')
    
    return render_template(f'{predicted_soil_type.lower().replace(" ", "")}.html')

def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.0

def predict_soil_type(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    confidence = np.max(prediction)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class, confidence

def download_blob_to_tmp(blob):
    """Download blob to temporary file and return file path."""
    import tempfile
    _, temp_local_filename = tempfile.mkstemp()
    blob.download_to_filename(temp_local_filename)
    return temp_local_filename



# this is the logic for crop prediction
# Define which columns are categorical for label encoding

# Load and prepare data
data = pd.read_csv('DataSet.csv')
categorical_columns = ['divisions', 'States']
encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    encoders[col] = encoder

# Get unique dropdown values from the dataset
states = data['States'].unique().tolist()
divisions = data['divisions'].unique().tolist()

@app.route('/')
def home():
    # Convert states and divisions for display in the dropdown
    state_labels = [encoder.inverse_transform([s])[0] for s in states]
    division_labels = [encoder.inverse_transform([d])[0] for d in divisions]
    return render_template('crop_prediction.html', states=state_labels, divisions=division_labels)

@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    state = request.form.get('States')
    division = request.form.get('divisions')
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rainfall = request.form.get('rainfall')

    if not all([state, division, temperature, humidity, ph, rainfall]):
        return render_template('crop_prediction.html', error="All fields are required.", states=states, divisions=divisions)

    try:
        temperature = float(temperature)
        humidity = float(humidity)
        ph = float(ph)
        rainfall = float(rainfall)
    except ValueError:
        return render_template('crop_prediction.html', error="Please enter valid numbers for temperature, humidity, pH, and rainfall.", states=states, divisions=divisions)

    # Encode the categorical data using the previously fitted LabelEncoders
    user_input = {
        'States': encoders['States'].transform([state])[0],
        'divisions': encoders['divisions'].transform([division])[0],
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }

    # Prepare data for model
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and predict using Decision Tree Classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    predicted_label = model.predict([list(user_input.values())])[0]

    # Render results along with the form for potential re-submission
    return render_template('crop_prediction.html', predicted_crop=predicted_label, accuracy=accuracy, states=states, divisions=divisions)

if __name__ == '__main__':
    app.run(debug=True)
    



if __name__ == '__main__':
    app.run(debug=True)
