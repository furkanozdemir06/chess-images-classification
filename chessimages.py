import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Chess Piece Classifier", page_icon="♟️", layout="centered")

# Same order the model was trained with (alphabetical folder names)
CLASSES = ["Queen", "Rook", "bishop", "knight", "pawn"]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("chess_model.keras")


model = load_model()

st.title("♟️ Chess Piece Classifier")
st.caption("Upload a photo of a chess piece and the model will tell you what it is.")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    x = np.expand_dims(np.array(img.resize((224, 224))) / 255.0, axis=0)
    probs = model.predict(x, verbose=0)[0]
    best = int(np.argmax(probs))

    col1, col2 = st.columns(2)
    col1.image(img, caption="Your image", use_container_width=True)
    col2.metric("Prediction", CLASSES[best].capitalize(), f"{probs[best] * 100:.1f}% confidence")

    chart = pd.DataFrame({"Piece": [c.capitalize() for c in CLASSES], "Probability": probs})
    st.plotly_chart(px.bar(chart, x="Piece", y="Probability", title="Class probabilities"),
                    use_container_width=True)

with st.expander("About the model"):
    st.write("MobileNetV2 (ImageNet weights) with a small classification head, "
             "trained on ~650 images of 5 chess pieces: queen, rook, bishop, knight and pawn.")