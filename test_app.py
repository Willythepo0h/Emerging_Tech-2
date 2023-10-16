# test_app.py
import app  # Import the main Streamlit application
from PIL import Image
import io

def test_app_title():
    # Test the title of the Streamlit app
    with app.st.echo():
        app.main()
    assert app.st.title == "Weather Classification Model"

def test_upload_and_predict():
    # Test image upload and prediction
    with app.st.echo():
        app.main()

    # Simulate image upload
    test_image = Image.open('Weather_girl.jpg')
    image_bytes = io.BytesIO()
    test_image.save(image_bytes, format="JPEG")
    app.st.file_uploader('Test Image', type=["jpg", "png"], value=image_bytes.getvalue())
    app.st.text("Please upload an image file")

    # Ensure the app displays the uploaded image
    assert app.st.image
    assert app.st.text("Please upload an image file")

    # Simulate image prediction
    prediction = app.import_and_predict(test_image, app.model)
    assert prediction

    # Ensure the prediction is displayed
    assert app.st.success("Prediction: ")
    assert app.st.write("Confidence Score: ")

def test_load_model():
    # Test loading the model
    model = app.load_model()
    assert model
