import numpy as np
import tensorflow as tf
from tensorflow import keras

# THIS CODE IS USED FOR EVALUATING EXISTING MODELS

# Labels for the evaluation data
labels = ["apple", "campfire", "diamond", "donut", "face", "fish", "hand", "house", "pizza", "t-shirt"]
# The different optimizing modes
modes = ["Regular", "Smoothing", "Rescaling", "Re_Smoothing"]

# Model to evaluate
modelPath = "testmodel1_500k"

# Information about the data image size
image_size = (255, 255)
img_height = 255
img_width = 255

# Number of images per category in the evaluation set. Used for iteration.
imgPerDir = 10


# Loading the model
model = keras.models.load_model(modelPath)


# Converting Labels for Models trained with 21 categories
def ConvertLabel(originalLabel):
    # No match case in Version 3.9 :((

    if originalLabel == 0:
        return 0
    elif originalLabel == 3:
        return 1
    elif originalLabel == 6:
        return 2
    elif originalLabel == 7:
        return 3
    elif originalLabel == 9:
        return 4
    elif originalLabel == 10:
        return 5
    elif originalLabel == 12:
        return 6
    elif originalLabel == 13:
        return 7
    elif originalLabel == 17:
        return 8
    elif originalLabel == 19:
        return 9

    return 10

# Iteration over the different modes
for k in range(4):
    # Counts the number of times when the prediction was correct
    correctCounter = 0
    # Counts the certainty of the model when it's made a correct prediction
    certainty = 0
    # Iteration over different categories

    for i in range(0, 10):
        # Iteration over different images within a category

        for j in range(1, imgPerDir+1):
            # Loading the image
            img = tf.keras.utils.load_img(
                "EvaluationData/" + modes[k] + "/" + labels[i] +"/" + labels[i] + " (" + str(j) + ")"+ ".png", target_size=(img_height, img_width)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Predicting the image
            predictions = model.predict(img_array)
            result = np.argmax(predictions)

            # This is only used for models which have learned more than 10 classes
            # since they can predict out of range of the labels-array
            #result = ConvertLabel(result)
            #if result > 9: continue

            # Check if the result matches the
            correct = result == i

            if correct:
                correctCounter += 1
                sumPrediction = 0
                for l in range(0, 10):
                    if predictions[0][l] >= 0.0:
                        sumPrediction += predictions[0][l]
                if sumPrediction > 0:
                    certainty += np.max(predictions[0]) / sumPrediction

            # Can be used for more detailed information
            #print(str(j) + ". I think it's: " + labels[result] + " with " + str(np.max(score)*100) + "%  " + str(correct))

    print(modes[k] + "->  Correct in: " + str(correctCounter) + "/" + str(imgPerDir * 10) + " | Confidence: " + str(certainty / correctCounter) + " | Accuracy: " + str(correctCounter / (imgPerDir * 10)))


