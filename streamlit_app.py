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

image_filename = "data/coco.jpg"
image = cv2.cvtColor(cv2.imread(filename=str(image_filename)), code=cv2.COLOR_BGR2RGB)

# Resize to MobileNet image shape.
input_image = cv2.resize(src=image, dsize=(224, 224))

# Reshape to model input shape.
input_image = np.expand_dims(input_image, 0)
plt.imshow(image);


# ## Do Inference
# [back to top ⬆️](#Table-of-contents:)
# 

# In[6]:


result_infer = compiled_model([input_image])[output_layer]
result_index = np.argmax(result_infer)


# In[11]:


from pathlib import Path

imagenet_filename = Path('data/imagenet_2012.txt')
imagenet_classes = imagenet_filename.read_text().splitlines()
#print(imagenet_classes)


# In[12]:


# The model description states that for this model, class 0 is a background.
# Therefore, a background must be added at the beginning of imagenet_classes.
imagenet_classes = ["background"] + imagenet_classes
imagenet_classes[result_index]


# In[ ]:




