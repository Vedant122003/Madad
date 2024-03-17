import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.0  # Normalize to [0,1]

def predict_with_threshold(model_path, image_path, threshold=0.99):
    model = load_model(model_path)
    processed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(processed_image)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction, axis=1)[0]

    if confidence < threshold:  
        return "Unknown or Non-Soil Image", confidence

    # Assuming your model was trained with categorical encoding
    class_names = ['Alluvial', 'Black', 'Clay', 'Red']  # Update as per your classes
    return class_names[predicted_class], confidence

if __name__ == "__main__":
    model_path = './best_weights.hdf5'  # Path to your HDF5 model file
    image_path =    r"C:\Users\vedan\Desktop\crop prediction\test2.jpg"  # Update 'your_image.jpg' to the specific image you want to predict

    predicted_class, confidence = predict_with_threshold(model_path, image_path)
    print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}')
