import tensorflowjs as tfjs
from tensorflow import keras

# Load the model
model = keras.models.load_model("testmodel1_1mio")
# Export the model
tfjs.converters.save_keras_model(model, "Model1JS")