import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflowjs as tfjs
from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils


model = keras.models.load_model("testmodel")

img = tf.keras.utils.load_img(
    "testingData/test4.png", target_size=(255, 255)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(predictions)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(np.argmax(score), 100 * np.max(score))
)