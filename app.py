import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st

model = tf.keras.models.load_model(r"best.hdf5")
label_dict={0:'Mask',1:'No Mask'}
def func(img):
  x = img_to_array(img)
  images = np.array([x],dtype="float32")
  classes = model.predict(images)
  result = label_dict[np.argmax(classes)]
  return result

def main():
  st.title("Mask classification")

  uploaded_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    
  if uploaded_file is not None:
     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
     image = cv2.imdecode(file_bytes, 1)
     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
     faces = face_cascade.detectMultiScale(image, 1.3, 5)
     for (x,y,w,h) in faces:
        image = image[y:y+h, x:x+w]
     result=func(image)
     st.write(result)

if __name__ == "__main__":
    main()

