import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model_cnn.hdf5')
  return model

def test_load_model():
    model = load_model()
    assert isinstance(model, tf.keras.models.Model)

# Define a test function for the import_and_predict function
def test_import_and_predict():
    # Mock image data for testing
    image_data = Image.new('RGB', (224, 224))
    model = load_model()
    num_classes = 2  # Replace with the actual number of classes in your model
    prediction = import_and_predict(image_data, model, num_classes)
    assert prediction.shape == (1, num_classes)

model=load_model()
st.write("""#Weather Classification Model""")

image = Image.open('Weather_girl.jpg')
st.image(image, caption='Weather Classification Model - John Willard S. Sucgang')

with st.container():
  col2 = st.columns((2,50,2))
  st.header("Model Outputs")
  st.info("""Rain and Shine""")
    
file=st.file_uploader("Choose a photo from computer",type=["jpg","png"])

def import_and_predict(image_data,model):
    size=(224,224)
    image=ImageOps.fit(image_data,size,Image.LANCZOS)
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
    
st.info("""Github Repository Link: https://github.com/Willythepo0h/Emerging_Tech-2""")
st.info("""Google Colab Link: https://colab.research.google.com/drive/1z8Q1byGelG2QqQRY66CjqP1ky4lM3IL_?usp=sharing""")
if __name__ == "__main__":
    main()
