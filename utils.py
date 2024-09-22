import openvino as ov
import cv2
import numpy as np
import matplotlib.pyplot as plt

core = ov.Core()
model = core.read_model(model='models/v3-small_224_1.0_float.xml')
compiled_model = core.compile_model(model = model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def preprocess(image, input_layer):
    N, input_channels, input_height, input_width = input_layer.shape
    resized_image = cv2.resize(image, (input_width, input_height))
    transposed_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image, 0)
    return input_image 

def predict_image(image, conf_threshold):
    input_image = preprocess(image, input_layer_face)
    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)
    imagenet_classes = imagenet_filename.read_text().splitlines()
    imagenet_classes = ["background"] + imagenet_classes
    imagenet_classes[result_index]
    return 
