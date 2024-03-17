from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import io
import base64
from flask import Flask, render_template



app = Flask(__name__)

users = {}
app.secret_key = 'e5ac358c-f0bf-11e5-9e39-d3b532c10a28'

# Assuming the model and class names are globally defined as before
model_path = r"C:\Users\vedan\Desktop\crop prediction\best_weights.hdf5"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    print(f"Model file not found at {model_path}")
class_names = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']

@app.route('/', methods=['GET'])
def welcome():
    return render_template('welcome.html')

@app.route('/login_signup', methods=['GET', 'POST'])

# Dummy user store
def login_signup():
    if request.method == 'POST':
        action = request.form['action']
        
        if action == 'signup':
            name = request.form.get('name')  # Get the name for signup
            email = request.form['email']
            password = request.form['password']
            
            if email in users:
                flash('Email already exists.')
            else:
                # Store hashed password along with name
                users[email] = {'name': name, 'password_hash': generate_password_hash(password)}
                flash('User successfully registered.')
                return redirect(url_for('index'))
        
        elif action == 'signin':
            email = request.form['email']
            password = request.form['password']
            user_info = users.get(email)
            
            if user_info and check_password_hash(user_info['password_hash'], password):
                # Assuming you have session management in place
                # session['user_email'] = email
                return redirect(url_for('index'))
            else:
                flash('Invalid email or password.')
    
    return render_template('login.html')


@app.route('/signout')
def signout():
    # Your sign-out logic here (e.g., clearing session data)
    return redirect(url_for('login_signup'))


@app.route('/index')

def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    threshold = 0.99  # Confidence threshold

    # The initial part of handling file upload/image data remains the same
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
    elif 'imageData' in request.form:
        encoded_data = request.form['imageData'].split(',')[1]
        image_data = base64.b64decode(encoded_data) 
        img = Image.open(io.BytesIO(image_data))
        image_path = os.path.join('uploads', 'captured_image.png')
        img.save(image_path)
    else:
        return 'No image provided', 400

    # Call the prediction function and check against the threshold
    predicted_soil_type, confidence = predict_soil_type(image_path)
    os.remove(image_path)  # Clean up after prediction

    if confidence < threshold:
        return render_template('index.html', error_message='low_confidence')

    # Redirect based on the prediction
    return render_template(f'{predicted_soil_type.lower().replace(" ", "")}.html')

def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.0

def predict_soil_type(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    confidence = np.max(prediction)  # Extract the maximum confidence value and  The predict method applies the model to the preprocessed image and returns the prediction
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class, confidence  # Return both class and confidence

if __name__ == '__main__':
    if not os.path.isdir('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
