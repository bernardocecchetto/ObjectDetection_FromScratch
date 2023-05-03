import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2


NUM_CLASSES = pd.read_csv('F:/ObjectDetection_FromScratch/data/annotations/train_annotations.csv')
NUM_CLASSES = len(NUM_CLASSES['class'].unique())

def ObjectDetection():


    input = tf.keras.layers.Input(shape=(224, 224, 3))

    base_model = VGG16(input_tensor = input, weights='imagenet', include_top=False, input_shape = (224, 224, 3))

    # for layer in base_model.layers:
    #     layer.trainable = False

    x = base_model.output
    # MLP
    x_mlp = tf.keras.layers.Flatten()(x)
    x_mlp = tf.keras.layers.Dense(256, activation="relu")(x_mlp)

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
