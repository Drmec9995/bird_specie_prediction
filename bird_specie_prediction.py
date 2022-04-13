import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as sl
from matplotlib import pyplot as plt
import pandas as pd


# loading model
model = load_model('C:/Users/ABDULBASIT/Desktop/bird_specie_prediction/bird_specie_prediction.h5')

# class name
class_name =['AMERICAN GOLDFINCH', 'BARN OWL', 'CARMINE BEE-EATER',
       'DOWNY WOODPECKER', 'EMPEROR PENGUIN', 'FLAMINGO']
sl.title('Bird specie prediction')
sl.markdown('Upload an image of a bird')

# uploading the leaf image...
bird_image = sl.file_uploader('choose an image...',type=['jpg','png','jpeg'])
submit = sl.button('predict')

# on predict button click

if submit:
    if bird_image is not None:
        # convert the file to an opencv image
        
        file_bytes = np.asarray(bytearray(bird_image.read()),dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes,1) 
        
        # displaying the image
        
        
        
        # Resizing the image
        opencv_image = cv2.resize(opencv_image,(256,256))
        
        # convert image to 4 dimensional image
        opencv_image.shape = (1,256,256,3)
         
         # making prediction...
        mid_val = 35
        y_pred = model.predict(opencv_image)
        score = tf.nn.softmax(y_pred)
        sl.write(score)
        score = score.numpy()
        score = score * 100.0
        chart_data = pd.DataFrame(
        score,
        columns=class_name)
        sl.bar_chart(chart_data)
        if np.any(max(score) > mid_val):
            
            sl.image(opencv_image,channels='RGB')
            sl.write(opencv_image.shape)
           
                
            result = class_name[np.argmax(y_pred)]
            
            sl.title(str('This is looks more of a ' + result))
    else:
        sl.title('upload an image')
    
        