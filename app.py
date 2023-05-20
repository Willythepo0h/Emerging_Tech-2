import streamlit as st
import tensorflow as tf
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model_cnn.hdf5')
  return model
model=load_model()
st.write("""
# Weather Classification Model"""
        )

image = Image.open('Weather_girl.jpg')
st.image(image, caption='Weather Classification Model - John Willard S. Sucgang')

with st.container():
  col2 = st.columns((2,50,2))
  st.header("Model Outputs")
  st.info("""Rain and Shine""")
    
file=st.file_uploader("Choose a photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(224,224)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Shine','Rain',]
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
