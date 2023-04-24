# ObjectDetection_FromScratch
This repository contains a study of object detection being builded from scratch, with all explanation of it


The following study it was developed using the PASCAL VOC Dataset (https://public.roboflow.com/object-detection/pascal-voc-2012)

It was based on pre-trained models from tensorflow connected to a Multi-Layer Perceptron, which has two outputs: a regression head (to predict the coordinates) and a classification head (to predict which class if the specific object located).


The training script it is located at: [src/train/train.py](src/train/train.py)

The architecture it is located at: [src/cnn/architecture.py](src/cnn/architecture.py)


If you want to reproduce the study, please, download the dataset, and submit the CSVs with the annotations inside [data/annotations](data/annotations) named  [train_annotations](data/annotations/train_annotations.csv) and [valid_annotations](data/annotations/valid_annotations.csv), the folder with the images inside [data/](data/) named Pascal VOC 2012.v1-raw.tensorflow.

To be able to reproduce the same results, please install the requirements inserted inside [requirements.txt](requirements.txt). The Python version used was 3.9.16.
