import tensorflow as tf
import numpy as np
import cv2
import os
import mtcnn
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st

model = tf.keras.models.load_model(r"best.hdf5")
label_dict={0:'Mask',1:'No Mask'}

def detect_face(ad):
  detector = mtcnn.MTCNN()
  return detector.detect_faces(ad)


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

  
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
     res = detect_face(image)
     if res:
        res = max(res, key = lambda b: b['box'][2] *b['box'][3])
        img, _, _ = get_face(img,res['box'])
        result=func(img)
        st.write(result)
     else:
        st.write("No face found")

if __name__ == "__main__":
    main()

