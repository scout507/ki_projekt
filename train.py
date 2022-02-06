import matplotlib.pyplot as plt
from tensorflow import keras


# THIS CODE IS USED FOR TRAINING EXISTING MODELS


# Path to the model
modelPath = "testmodel1_500k"

# Data size
image_size = (255, 255)
img_height = 255
img_width = 255
batch_size = 32

# Create training and validation dataset
train_ds = keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Information about the different classes
class_names = train_ds.class_names
num_classes = len(class_names)

# Number of training epochs
epochs = 2

# Loading the model
model = keras.models.load_model(modelPath)


# Training
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
model.save("TestModel1_500k")

# Data for visualization
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)


# Plotting the training results
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


