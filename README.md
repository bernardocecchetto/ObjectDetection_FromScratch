# ObjectDetection_FromScratch
This repository contains a study of object detection being builded from scratch, with all explanation of it


# The following study it was developed using the PASCAL VOC Dataset (https://public.roboflow.com/object-detection/pascal-voc-2012)

## It was based on pre-trained models from tensorflow connected to a Multi-Layer Perceptron, which has two outputs: a regression head (to predict the coordinates) and a classification head (to predict which class if the specific object located).


## The training script it is located at: src/train/train.py

## The architecture it is located at: src/cnn/architecture.py