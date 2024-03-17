import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

model_path = r"C:\Users\vedan\Desktop\crop prediction\best_weights.hdf5"

# Load the trained model
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.0  # Ensure you scale the images in the same way as during training

# Function to predict the soil type
def predict_soil_type(image_path):
    processed_image = preprocess_image(image_path)
    
    # Adjust class names as per your dataset labels
    class_names = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']
    
    # Assuming your model outputs one-hot encoded vectors
    prediction = model.predict(processed_image)
    
    # Find the index of the highest predicted score
    predicted_class = class_names[np.argmax(prediction)]
    
    return predicted_class

# Use the specific image path
image_path = r"C:\Users\vedan\Desktop\Copy of image2.jpeg"

predicted_soil_type = predict_soil_type(image_path)

print(f"The predicted soil type is: {predicted_soil_type}")
