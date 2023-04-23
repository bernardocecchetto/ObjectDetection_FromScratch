import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import shutil
import tensorflow as tf
import typing as ty
import json
import cv2 as cv2
import logging


"""
This code will save each image in a folder of train/validation, coresponding to your class, so we can use load_dataset from tensorflow, which is easier
"""

train_csv = pd.read_csv('F:/ObjectDetection_FromScratch/data/annotations/train_annotations.csv', delimiter=',')
valid_csv = pd.read_csv('F:/ObjectDetection_FromScratch/data/annotations/valid_annotations.csv', delimiter=',')

# Opening JSON file
f = open('F:/ObjectDetection_FromScratch/data/annotations/label_map.json')
# returns JSON object as  a dictionary
json_labels = json.load(f)


# definition of features dtype
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def generate_tfrecords(dataframe: pd.DataFrame(), data_type: ty.AnyStr, dict_labels: ty.Dict):
   

   print("Running in the {0}".format(data_type))
   for _, row in tqdm(dataframe.iterrows(), total = len(dataframe)):
        # storing features
        img_name = row['filename']
        label_name = row['class']
        label_num = int(dict_labels[label_name])
        xmin, xmax, ymin, ymax = row['xmin'], row['xmax'], row['ymin'], row['ymax']

        feature = {
            'xmin': _int64_feature(xmin), # saving xmin
            'xmax': _int64_feature(xmax), # saving xmax
            'ymin': _int64_feature(ymin), # saving ymin
            'ymax': _int64_feature(ymax), # saving xmax
            'label': _int64_feature(label_num), # saving label
        }
        tf_record = tf.train.Example(features=tf.train.Features(feature=feature))

        if not os.path.isdir(f'F:/ObjectDetection_FromScratch/data/{data_type}'):
            os.makedirs(f'F:/ObjectDetection_FromScratch/data/{data_type}')

        with tf.io.TFRecordWriter(f'F:/ObjectDetection_FromScratch/data/{data_type}/' + img_name + '_.tfrecords') as writer:
            writer.write(tf_record.SerializeToString())


def main():
    generate_tfrecords(train_csv, "train", json_labels)
    generate_tfrecords(valid_csv, "valid", json_labels)

if __name__ == "__main__":
    main()