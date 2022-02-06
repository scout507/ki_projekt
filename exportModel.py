import tensorflowjs as tfjs
from tensorflow import keras

# THIS IS USED FOR EXPORTING EXISTING MODELS  

# Load the model
model = keras.models.load_model("testmodel1_1mio")
# Export the model
tfjs.converters.save_keras_model(model, "Model1JS")
