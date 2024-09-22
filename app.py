#!/usr/bin/env python
# coding: utf-8

# # Hello Image Classification
# 
# This basic introduction to OpenVINO™ shows how to do inference with an image classification model.
# 
# A pre-trained [MobileNetV3 model](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/mobilenet-v3-small-1.0-224-tf/README.md) from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/) is used in this tutorial. For more information about how OpenVINO IR models are created, refer to the [TensorFlow to OpenVINO](../tensorflow-classification-to-openvino/tensorflow-classification-to-openvino.ipynb) tutorial.
# 
# 
# #### Table of contents:
# 
# - [Imports](#Imports)
# - [Download the Model and data samples](#Download-the-Model-and-data-samples)
# - [Select inference device](#Select-inference-device)
# - [Load the Model](#Load-the-Model)
# - [Load an Image](#Load-an-Image)
# - [Do Inference](#Do-Inference)
# 
# 
# ### Installation Instructions
# 
# This is a self-contained example that relies solely on its own code.
# 
# We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
# For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).
# 
# <img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/hello-world/hello-world.ipynb" />
# 

# ## Imports
# [back to top ⬆️](#Table-of-contents:)
# 

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov


# In[2]:


# Version Check
import sys, matplotlib
import ipywidgets as widgets
print("python==",sys.version)
print("numpy==",np.__version__)
print("opencv-python==",cv2.__version__)
print("matplotlib==",matplotlib.__version__)
print("openvino==",ov.__version__)
print("device_widget==",widgets.__version__)


# ## Select inference device
# [back to top ⬆️](#Table-of-contents:)
# 
# select device from dropdown list for running inference using OpenVINO

# In[3]:


core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False,
)

device


# ## Load the Model
# [back to top ⬆️](#Table-of-contents:)
# 

# In[4]:


core = ov.Core()

model_xml_path="./artifacts/v3-small_224_1.0_float.xml"

model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device.value)

output_layer = compiled_model.output(0)


# ## Load an Image
# [back to top ⬆️](#Table-of-contents:)
# 

# In[5]:


image_filename = "data/coco.jpg"
# The MobileNet model expects images in RGB format.
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




