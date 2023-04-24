import tensorflow as tf
from tensorflow.keras.utils import Sequence

import sys

sys.path.append(".")
from src.cnn.architecture import ObjectDetection
import glob
import numpy as np
from PIL import Image
import io
import os
import cv2 as cv2
import typing as ty
from tqdm import tqdm
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import json
import pandas as pd
import imagesize
from sklearn import preprocessing
import re
from tensorflow.keras import backend as K

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

BATCH_SIZE = 8
EPOCHS = 100
CLASSES = 20
IMG_DIR = 'F:/ObjectDetection_FromScratch/data/Pascal VOC 2012.v1-raw.tensorflow/'
NUM_CLASSES = pd.read_csv('F:/ObjectDetection_FromScratch/data/annotations/train_annotations.csv')
NUM_CLASSES = len(NUM_CLASSES['class'].unique())

def toPercentage(dimx, dimy, x1, x2, y1, y2):
    h, w, c = dimx, dimy, 3
    x1p = x1 / w
    x2p = x2 / w
    y1p = y1 / h
    y2p = y2 / h
    return x1p, x2p, y1p, y2p


def toImCoord(x1p, x2p, y1p, y2p):
    h, w, c = 224, 224, 3
    x1 = x1p * w
    x2 = x2p * w
    y1 = y1p * h
    y2 = y2p * h
    return x1, x2, y1, y2


def load_data(data_type: ty.AnyStr = None, le:preprocessing.LabelEncoder = None):

    df = pd.read_csv(f'F:/ObjectDetection_FromScratch/data/annotations/{data_type}_annotations.csv')
    df['filename'] = df['filename'].apply(lambda x: IMG_DIR + data_type + '/' + x)
    

    for _, row in df.iterrows():
        # recalculating the coordinates after resizing the image
        xminp, xmaxp, yminp, ymaxp = toPercentage(
            row['width'], row['height'], row['xmin'], row['xmax'], row['ymin'], row['ymax'])
        xmin, xmax, ymin, ymax = toImCoord(xminp, xmaxp, yminp, ymaxp)

        row['xmin'], row['xmax'], row['ymin'], row['ymax'] = xmin, xmax, ymin, ymax
    if data_type == 'train':
        le = preprocessing.LabelEncoder()
        le.fit(df['class'].values)
        df['class'] = le.transform(df['class'])
    else:
        df['class'] = le.transform(df['class'])


    df.drop(columns=['width', 'height'], inplace=True)
    df.rename(columns={'class': 'label'}, inplace=True)

    return df, le


def adapt(generator):
    def new_generator():
        for img, (label, xmin, xmax, ymin, ymax) in generator:
            x = img
            y = {
                "cls": np.eye(NUM_CLASSES)[label],
                "reg": np.stack([xmin, xmax, ymin, ymax], axis=1),
            }
            yield x, y
    return new_generator


def f1_score(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def compile_model(model, lr: float = 1e-4):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    losses = {
        "cls": tf.keras.losses.CategoricalCrossentropy(),
        "reg": tf.keras.losses.MeanSquaredError(),
    }

    metrics = {
        "cls": f1_score,
        "reg": tf.keras.metrics.MeanSquaredError(),
    }

    model.compile(optimizer, losses, metrics)

    return model


def main():

    df_train, transformer = load_data('train')
    df_valid, _ = load_data('valid', transformer)

    image_data_gen_args = dict(
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="multi_output",
        x_col="filename",
        y_col=["label", "xmin", "xmax", "ymin", "ymax"],
    )

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255.0
    ).flow_from_dataframe(
        df_train,
        directory="F:/ObjectDetection_FromScratch/data/train",
        **image_data_gen_args,
    )

    valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255.0
    ).flow_from_dataframe(
        df_valid,
        directory="F:/ObjectDetection_FromScratch/data/valid",
        **image_data_gen_args,
    )

    output_signature = (
        tf.TensorSpec(shape=(None, 224, 224, 3)),
        {
            "cls": tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.int32),
            "reg": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        },
    )

    
    train_ds = tf.data.Dataset.from_generator(
        adapt(train_gen), output_signature=output_signature
    )
    valid_ds = tf.data.Dataset.from_generator(
        adapt(valid_gen), output_signature=output_signature
    )


    for img, (label, xmin, xmax, ymin, ymax) in valid_gen:
        print("Image shape: ")
        print(img.shape, '\n')

        print("Labels to classification: ")
        print(label, '\n')

        print("Regression: ")
        print("X[0]: ", xmin)
        print("X[1]: ", xmax)
        print("Y[0]: ", ymin)
        print("Y[1]: ", ymax)
        break


    model = ObjectDetection()
    model = compile_model(model)

    wandb.init(
        # set the wandb project where this run will be logged
        project="objectDetection",
    )

    history = model.fit(
        train_ds,
        steps_per_epoch=(len(df_train) // BATCH_SIZE),
        validation_data=valid_ds,
        validation_steps=1,
        epochs=EPOCHS,
        callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")],
    )

    wandb.finish()


if __name__ == "__main__":
    main()
