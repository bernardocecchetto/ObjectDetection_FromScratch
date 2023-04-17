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

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

BATCH_SIZE = 32
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

def toPercentage(img_orig, x1,x2,y1,y2):
    h,w,c = np.shape(img_orig)
    x1p = x1 / w
    x2p = x2 / w
    y1p = y1 / h
    y2p = y2 / h
    return x1p, x2p, y1p, y2p

def toImCoord(img_resized, x1p,x2p,y1p,y2p):
    h,w,c = np.shape(img_resized)
    x1 = x1p * w
    x2 = x2p * w
    y1 = y1p * h
    y2 = y2p * h        
    return x1, x2, y1, y2



def load_data(data_type: ty.AnyStr):
    tfrecords = glob.glob(f"F:/ObjectDetection_FromScratch/data/{data_type}/*tfrecords")
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    Y = []
    img = np.zeros(((len(tfrecords), 224, 224, 3)))

    if not os.path.exists(
        f"F:/ObjectDetection_FromScratch/data/Pascal VOC 2012.v1-raw.tensorflow_processed/{data_type}/"
    ):
        os.makedirs(
            f"F:/ObjectDetection_FromScratch/data/Pascal VOC 2012.v1-raw.tensorflow_processed/{data_type}"
        )
        new_coords = {}
        for idx, sample in enumerate(tqdm(tfrecords)):
            raw_image_dataset = tf.data.TFRecordDataset(sample)
            img_filename = sample.split("\\")[-1].split("_.tfrecords")[0]

            image = tf.keras.utils.load_img(
                f"F:/ObjectDetection_FromScratch/data/Pascal VOC 2012.v1-raw.tensorflow/{data_type}/"
                + img_filename,
                target_size=(224, 224),
            )
            image = tf.keras.utils.img_to_array(image)
            image = normalization_layer(image)
            tf.keras.utils.save_img(
                f"F:/ObjectDetection_FromScratch/data/Pascal VOC 2012.v1-raw.tensorflow_processed/{data_type}/{img_filename}",
                image,
                data_format="channels_last",
            )

            img[idx, :, :] = image
            # Create a dictionary describing the features.
            parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
            for image_features in parsed_image_dataset:
                raw_xmin, raw_xmax, raw_ymin, raw_ymax = (
                    image_features["xmin"],
                    image_features["xmax"],
                    image_features["ymin"],
                    image_features["ymax"],
                )
                label = image_features["label"]

            raw_img = cv2.imread(f"F:/ObjectDetection_FromScratch/data/Pascal VOC 2012.v1-raw.tensorflow/{data_type}/{img_filename}")

            # recalculating the coordinates after resizing the image
            xminp, xmaxp, yminp, ymaxp = toPercentage(raw_img, raw_xmin, raw_xmax, raw_ymin, raw_ymax)
            xmin, xmax, ymin, ymax = toImCoord(image, xminp, xmaxp, yminp, ymaxp)


            # saving the new coordinates in the first run
            new_coords[img_filename] = [xmin, xmax, ymin, ymax]

            box = [xmin, xmax, ymin, ymax]
            box = np.asarray(box, dtype=float)
            labels = np.append(box, label)

            Y.append(labels)

        with open(f"F:/ObjectDetection_FromScratch/data/annotations/new_coords.json", "w") as pbfile:
            json.dump(new_coords, pbfile)

        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        labels_tensor = tf.convert_to_tensor(Y, dtype=tf.float32)
        result = tf.data.Dataset.from_tensor_slices((img_tensor, labels_tensor))

    else:
        Y = []
        # Opening JSON file
        f = open('F:/ObjectDetection_FromScratch/data/annotations/new_coords.json')
        # returns JSON object as  a dictionary
        json_coords = json.load(f)

        for idx, sample in enumerate(tqdm(tfrecords)):
            raw_image_dataset = tf.data.TFRecordDataset(sample)
            img_filename = sample.split("\\")[-1].split("_.tfrecords")[0]

            image = tf.keras.utils.load_img(
                f"F:/ObjectDetection_FromScratch/data/Pascal VOC 2012.v1-raw.tensorflow_processed/{data_type}/"
                + img_filename,
                target_size=(224, 224),
            )

            img[idx, :, :] = image
            # Create a dictionary describing the features.
            parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

            for image_features in parsed_image_dataset:
                label = image_features["label"]

            box = json_coords[img_filename]
            box = np.asarray(box, dtype=float)

            labels = np.append(box, label)
            Y.append(labels)


        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        labels_tensor = tf.convert_to_tensor(Y, dtype=tf.float32)
        result = tf.data.Dataset.from_tensor_slices((img_tensor, labels_tensor))

    return result


def format_instance(image, label):
    return image, (
        tf.one_hot(int(label[4]), CLASSES),
        [label[0], label[1], label[2], label[3]],
    )


def tune_training_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.repeat()  # The dataset be repeated indefinitely.
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def tune_validation_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(len(dataset) // 4)
    dataset = dataset.repeat()
    return dataset


def main():
    raw_train_ds = load_data("train")
    raw_valid_ds = load_data("valid")

    train_ds = tune_training_ds(raw_train_ds)
    valid_ds = tune_validation_ds(raw_valid_ds)

    model = ObjectDetection(tf.keras.layers.Input(shape=(224, 224, 3)))

    wandb.init(
        # set the wandb project where this run will be logged
        project="objectDetection",
        # track hyperparameters and run metadata with wandb.config
        config={
            "classifier_head_loss": "categorical_crossentropy",
            "regressor_head_loss": "mse",
            "classifier_head_metric": "accuracy",
            "regressor_head_metric": "mse",
        },
    )
    config = wandb.config

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss={
            "classifier_head": config.classifier_head_loss,
            "regressor_head": config.regressor_head_loss,
        },
        metrics={
            "classifier_head": config.classifier_head_metric,
            "regressor_head": config.regressor_head_metric,
        },
    )

    history = model.fit(
        train_ds,
        steps_per_epoch=(len(raw_train_ds) // BATCH_SIZE),
        validation_data=valid_ds,
        validation_steps=1,
        epochs=EPOCHS,
        callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")],
    )

    wandb.finish()


if __name__ == "__main__":
    main()


# model.fit(X_train, [y_train,y_train_class], epochs=150, batch_size=32, verbose=2)
