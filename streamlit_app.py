import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov

st.set_page_config(
    page_title="Hello Image Classification",
    page_icon=":sun_with_face:",
    layout="centered",
    initial_sidebar_state="expanded",)

st.title("Hello Image Classification :sun_with_face:")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20))/100

input = None 
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))
    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold = conf_threshold)
        st.image(visualized_image, channels = "BGR")
    else: 
        st.image("data/coco.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image." )
        
if source_radio == "WEBCAM":
    image = camera_input_live()
    uploaded_image = PIL.Image.open(image)
    uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image), cv2.COLOR_RGB2BGR)
    visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold = conf_threshold) 
    st.image(visualized_image, channels = "BGR")
