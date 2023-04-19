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
import re
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

BATCH_SIZE = 4
EPOCHS = 100
CLASSES = 20


def _parse_image_function(example_proto):
    image_feature_description = {
        "xmin": tf.io.FixedLenFeature([], tf.int64),
        "xmax": tf.io.FixedLenFeature([], tf.int64),
        "ymin": tf.io.FixedLenFeature([], tf.int64),
        "ymax": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


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


def load_data(data_type: ty.AnyStr):
    tfrecords = glob.glob(f"F:/ObjectDetection_FromScratch/data/{data_type}/*tfrecords")


    dict_data = {}
    for _, sample in enumerate(tqdm(tfrecords)):
        raw_image_dataset = tf.data.TFRecordDataset(sample)
        img_filename = sample.split("\\")[-1].split("_.tfrecords")[0]

        # Create a dictionary describing the features.
        parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
        for image_features in parsed_image_dataset:
            raw_xmin, raw_xmax, raw_ymin, raw_ymax = (
                float(image_features["xmin"]),
                float(image_features["xmax"]),
                float(image_features["ymin"]),
                float(image_features["ymax"]),
            )
            label = image_features["label"]

        x, y = imagesize.get(f"F:/ObjectDetection_FromScratch/data/Pascal VOC 2012.v1-raw.tensorflow/{data_type}/{img_filename}")
        
        # recalculating the coordinates after resizing the image
        xminp, xmaxp, yminp, ymaxp = toPercentage(
            x, y, raw_xmin, raw_xmax, raw_ymin, raw_ymax
        )
        xmin, xmax, ymin, ymax = toImCoord(xminp, xmaxp, yminp, ymaxp)

        box = [xmin, xmax, ymin, ymax]
        box = np.asarray(box, dtype=np.float32)

        dict_data[img_filename] = {'label': label, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax':ymax}

        df = pd.DataFrame().from_dict(dict_data, orient='index')
        df.to_csv(f"F:/ObjectDetection_FromScratch/data/{data_type}.csv")
    return df

def adapt(generator):
    def new_generator():
        for img, (label, xmin, xmax, ymin, ymax) in generator:
            x = img
            y = {
                'reg': np.stack([xmin, xmax, ymin, ymax], axis=1),
                'cls': np.eye(20)[label]

            }
            yield x, y
        return new_generator
            

def compile_model(model, optimizer=tf.keras.optimizers.Adam(), lr: float = 1e-4):
    optimizer=optimizer(lr)
    losses = {
        'cls': tf.keras.losses.CategoricalCrossentropy(),
        'reg': tf.keras.losses.MeanSquaredError(),
    }

    metrics = {
        'cls': tf.keras.metrics.Accuracy(),
        'reg': tf.keras.metrics.IoU(),
    }

    model.compie(optimizer, losses, metrics)

    return model


def main():
    if not os.path.exists('F:/ObjectDetection_FromScratch/data/train.csv') and not os.path.exists('F:/ObjectDetection_FromScratch/data/valid.csv'):
        df_train = load_data('train')
        df_valid = load_data('valid')

    elif not os.path.exists('F:/ObjectDetection_FromScratch/data/train.csv') and os.path.exists('F:/ObjectDetection_FromScratch/data/valid.csv'):
        df_train = load_data('train')

    elif os.path.exists('F:/ObjectDetection_FromScratch/data/train.csv') and not os.path.exists('F:/ObjectDetection_FromScratch/data/valid.csv'):
        df_valid = load_data('valid')

    else:
        df_train = pd.read_csv('F:/ObjectDetection_FromScratch/data/train.csv')
        df_valid = pd.read_csv('F:/ObjectDetection_FromScratch/data/valid.csv')


    image_data_gen_args = dict(
        target_size =(224, 224),
        batch_size = BATCH_SIZE,
        class_mode = 'multi_output',
        x_col = 'filename',
        y_col = ['label', 'xmin', 'xmax', 'ymin', 'ymax']
    )

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255.
    ).flow_from_dataframe(df_train, directory='F:/ObjectDetection_FromScratch/data/train', **image_data_gen_args)

    valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255.
    ).flow_from_dataframe(df_valid, directory='F:/ObjectDetection_FromScratch/data/valid', **image_data_gen_args)

    output_signature = (
        tf.TensorShape(shape = (None, 224, 224, 3)),

        {
            'cls': tf.TensorShape(shape=(None, 20), dtype = tf.int32),
            'reg': tf.TensorShape(shape=(None, 4), dtype = tf.float32)

        }
    )

    train_ds = tf.data.Dataset.from_generator(
        adapt(train_gen),
        output_signature=output_signature
    )
    valid_ds = tf.data.Dataset.from_generator(
        adapt(valid_gen),
        output_signature=output_signature
    )


    model = ObjectDetection()

    wandb.init(
        # set the wandb project where this run will be logged
        project="objectDetection",
        # track hyperparameters and run metadata with wandb.config
        #     
         )

    model = compile_model(model)

    history = model.fit(
        train_ds,
        steps_per_epoch=(len(num_filest_train) // BATCH_SIZE),
        validation_data=valid_ds,
        validation_steps=1,
        epochs=EPOCHS,
        callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")],
    )

    wandb.finish()


if __name__ == "__main__":
    main()
