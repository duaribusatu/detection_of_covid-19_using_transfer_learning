import os
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import *
from keras.models import load_model
import tensorflow_hub as hub
import time

fig = plt.figure()
st.title('COVID-19 Detection Image')
st.header('Digunakan sebagai prototipe deteksi covid-19 melalui image CXR')
st.markdown("Prediction : (Covid-19 or Normal) | Made by: Dandi Septiandi (1907112680)")

# Upload file dan tampilan web app
def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Predict")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict_image(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)

# Function to preprocess image before making prediction
def predict_image(image):
    # Upload model
    mymodel = 'model/EFv12400_covid_detection_model.h5'
    classifier_model = load_model(mymodel, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    # Resize image
    image_process = image.resize((224, 224))
    # Convert image to array and normalize
    image_array = tf.keras.preprocessing.image.img_to_array(image_process)
    image_array = image_array / 255.0
    # Expand dimensions
    image_array = np.expand_dims(image_array, axis=0)
    # Class
    class_names = {0: 'Normal', 1: 'Covid-19'}
    predictions = classifier_model.predict(image_array)
    if np.max(predictions[0]) > 0.5:
        scores = np.max(predictions[0]) * 100 
        predicted_class = class_names[np.argmax(predictions[0])]
    else:
        scores = (1.0 - np.max(predictions[0])) * 100
        predicted_class = class_names[np.argmax(predictions[0])]
    result = f"{predicted_class} with a {(scores).round(2)} % confidence."
    return result

if __name__ == "__main__":
    main()
