import streamlit as st
import requests
from PIL import Image

# Hacky way to center images as customizable layout is currently not supported by Streamlit
# (source: https://discuss.streamlit.io/t/how-to-center-images-latex-header-title-etc/1946/4)
with open("style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

st.title("Judging a Book by Its Cover")

uploaded_file = st.file_uploader(
    "Upload a book cover (supported formats: jpeg, jpg, png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    # Show uploaded image
    image = Image.open(uploaded_file)
    image_width = min(image.size[1], 400)
    st.image(image, caption="Your Book Cover", width=image_width)

    # Query for model prediction
    file = {"file": uploaded_file.getvalue()}
    response = requests.post("http://backend:8080/judge", files=file)

    if response.status_code != requests.codes.ok:
        st.markdown(
            "<div style='text-align: center;'>Invalid format - image must be in RGB mode!</div>",
            unsafe_allow_html=True,
        )

    else:
        predicted_book_rating = response.json()["predicted_book_rating"]
        st.markdown(
            f"<div style='text-align: center;'>Predicted Rating: <b>{predicted_book_rating}</b></div>",
            unsafe_allow_html=True,
        )
