import tensorflow as tf
import sys
sys.path.append('.')
from src.cnn.architecture import ObjectDetection
import glob
import numpy as np
from PIL import Image
import io
import typing as ty
from tqdm import tqdm

def _parse_image_function(example_proto):
    image_feature_description = {
    'xmin': tf.io.FixedLenFeature([], tf.int64),
    'xmax': tf.io.FixedLenFeature([], tf.int64),
    'ymin': tf.io.FixedLenFeature([], tf.int64),
    'ymax': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


def load_data(data_type: ty.AnyStr):
    tfrecords = glob.glob(f'F:/ObjectDetection_FromScratch/data/{data_type}/*tfrecords')
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    box = []
    labels = []
    img=np.zeros(((len(tfrecords),224,224,3)))

    for idx, sample in enumerate(tqdm(tfrecords)):
        raw_image_dataset = tf.data.TFRecordDataset(sample)
        img_filename =  sample.split('\\')[-1].split('_.tfrecords')[0]

        image = tf.keras.utils.load_img(f'F:/ObjectDetection_FromScratch/data/Pascal VOC 2012.v1-raw.tensorflow/{data_type}/' + img_filename, target_size=(224,224))
        image = tf.keras.utils.img_to_array(image)
        image = normalization_layer(image)
        img[idx, :, :] = image
        # Create a dictionary describing the features.
        parsed_image_dataset = raw_image_dataset.map(_parse_image_function)


        for image_features in parsed_image_dataset:            

            xmin, xmax, ymin, ymax = image_features['xmin'], image_features['xmax'], image_features['ymin'], image_features['ymax']
            box.append([xmin, xmax, ymin, ymax])
            label = image_features['label']
            labels.append(label)


    return img, box, labels


def main():

    img_train, box_train, labels_train = load_data('train')
    img_valid, box_valid, labels_valid = load_data('valid')

    model = ObjectDetection(tf.keras.layers.Input(shape=(224, 224, 3)))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-5), 
        loss = {'classifier_head' : 'categorical_crossentropy', 'regressor_head' : 'mse' }, 
        metrics = {'classifier_head' : 'accuracy', 'regressor_head' : 'mse' })
        
    EPOCHS = 100
    BATCH_SIZE = 32

    history = model.fit(img_train,[labels_train, box_train],
                        validation_data=[img_valid, [labels_valid, box_valid]], batch_size = BATCH_SIZE, 
                        epochs=EPOCHS)




if __name__ == "__main__":
    main()




#model.fit(X_train, [y_train,y_train_class], epochs=150, batch_size=32, verbose=2)