import streamlit as st
import requests
from PIL import Image

st.title("Judging a Book by Its Cover")

uploaded_file = st.file_uploader(
    "Upload a book cover (supported formats: jpeg, jpg, png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    column1, column2, column3, column4 = st.beta_columns(4)

    # Show uploaded image
    image = Image.open(uploaded_file)
    image_width = min(image.size[1], 400)
    column2.image(image, caption="Your Book Cover", width=image_width)

    # Query for model prediction
    file = {"file": uploaded_file.getvalue()}
    response = requests.post("http://backend:8080/judge", files=file)

    if response.status_code != requests.codes.ok:
        column3.markdown("Invalid format - image must be in RGB mode!")

    else:
        predicted_book_rating = response.json()["predicted_book_rating"]
        column3.markdown(
            f"<br/><br/>Predicted Rating: <b>{predicted_book_rating}</b>",
            unsafe_allow_html=True,
        )
