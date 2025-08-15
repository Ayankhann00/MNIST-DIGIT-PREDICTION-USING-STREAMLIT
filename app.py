import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.keras")

st.title("✏️ Draw a Digit & Let the CNN Guess!")
st.write("Draw a digit (0–9) in the box below, then click 'Predict'.")

# Create a drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
    
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        
    
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img).astype("float32") / 255
        img_array = img_array.reshape(1, 28, 28, 1)


        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100

        st.write(f"**Prediction:** {predicted_digit}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.bar_chart(prediction[0])
    else:
        st.warning("Please draw something before predicting!")

