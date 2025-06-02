import streamlit as st
import os
from PIL import Image
import main
import time

st.set_page_config(page_title="eigenface", layout="wide")
st.title("Face Recognition using EigenFace")

st.sidebar.header("Settings")

datasetPath = st.sidebar.text_input("Dataset Folder Path", "")

testImageUpload = st.sidebar.file_uploader("Upload Test Image", type=["jpg", "jpeg", "png"])

thresholdValue = st.sidebar.slider(
    "Recognition Threshold (Euclidean Distance)", 0, 1000000, 500000
)

if st.sidebar.button("Start Recognition"):
    if datasetPath == "" or not os.path.isdir(datasetPath):
        st.error("Please input a valid dataset folder path!")
    elif testImageUpload is None:
        st.error("Please upload a test image!")
    else:
        with st.spinner("Processing..."):
            # Simpan test image sementara
            testImagePath = "temp_test_image.jpg"
            with open(testImagePath, "wb") as f:
                f.write(testImageUpload.read())

            startTime = time.time()
            matchedPath, matchPercentage, minDistance = main.run(datasetPath, testImagePath, thresholdValue)
            execTime = round(time.time() - startTime, 3)

            st.success(f"Recognition completed in {execTime} seconds")
            st.write(f"Threshold used: `{thresholdValue}`")
            st.write(f"Minimum Euclidean Distance: `{minDistance:.2f}`")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Test Image")
                st.image(Image.open(testImagePath), width=300)

            with col2:
                st.subheader("Closest Match")
                st.image(Image.open(matchedPath), width=300)

            if matchPercentage == 0:
                st.error("No matching face found!")
            else:
                st.info(f"Accuracy Percentage: {matchPercentage:.2f}%")
