import tensorflow as tf
import pandas as pd

NUM_CLASSES = pd.read_csv('F:/ObjectDetection_FromScratch/data/annotations/train_annotations.csv')
NUM_CLASSES = len(NUM_CLASSES['class'].unique())

def ObjectDetection():
    input = tf.keras.layers.Input(shape=(224, 224, 3))
    # convolutional layers 1
    x = tf.keras.layers.Conv2D(
        16, kernel_size=3, activation="relu", input_shape=(224, 224, 3)
    )(input)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    # convolutional layers 2
    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    # convolutional layers 3
    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    # MLP
    x_mlp = tf.keras.layers.Flatten()(x)
    x_mlp = tf.keras.layers.Dense(64, activation="relu")(x_mlp)

    # classification output
    x_class = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classifier_head")(
        x_mlp
    )

    # regression layer, to the coordinates
    x_reg = tf.keras.layers.Dense(4, name="regressor_head")(x_mlp)

    model = tf.keras.Model(
        inputs=input, outputs={"cls": x_class, "reg": x_reg}, name="object_location"
    )

    return model
