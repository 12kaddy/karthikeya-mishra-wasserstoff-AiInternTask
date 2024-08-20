# streamlit_app/app.py
import streamlit as st
from PIL import Image

st.title("AI Pipeline: Image Segmentation and Analysis")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image (call segmentation, identification, etc.)
    # Display segmented objects, descriptions, extracted text, and summaries

    if st.button("Show Final Output"):
        # Display final annotated image and summary table
        st.image('data/output/final_image.png')
        st.write(pd.read_csv('data/output/summary_table.csv'))
        