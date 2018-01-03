import tensorflow as tf
from PIL import Image
import numpy as np
import os
from random import shuffle


class Config:
    TRAIN_PATH = './datasets/train'
    DEV_PATH = './datasets/validation'
    TEST_PATH = './datasets/test'
    TARGET_DIR = 'apples'
    OTHER_DIR = 'others'

    TFRECORD_TRAIN = TRAIN_PATH + '/train.tfrecord'
    TFRECORD_DEV = DEV_PATH + '/dev.tfrecord'
    TFRECORD_TEST = TEST_PATH + '/test.tfrecord'

    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]
    IMAGE_SHAPE = [IMAGE_HEIGHT, IMAGE_WIDTH, 3]
    IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * 3

    LEAKY_MODEL_PATH = './models/leaky_model/leaky_model.ckpt'
    NORMAL_MODEL_PATH = './models/non_leaky_model/non_leaky_model.ckpt'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_binary_image(filename, size=None):
    """
    Reads image from specified file.
    @params:
        filename: Path to file to be readed
        size: To resize the image [width, height]. Default: None -> Do Not resize the image.
    @returns
        shape: shape of image in form of bytes
        img: Image in form of bytes
    """
    img = Image.open(filename)

    if size is not None:
        img = img.resize(size)

    img = np.asarray(img, np.uint8)
    if len(img.shape) < 3:
        print('File Removed: ' + filename)
        return None, None

    if img.shape[2] > 3:
        img = img[:, :, :3]

    shape = np.array(img.shape, np.int32)
        
    return shape.tobytes(), img.tobytes()


def write_to_tfrecord(labels, shapes, binary_imgs, tfrecord_file):
    """
    Take a bunch of images and write features to tfrecord_file
    @params
        labels: List of labels
        shapes: List of shapes of Images in form of bytes
        binary_imgs: List of Images in form of bytes
        tfrecord_file: Name of tfrecord_file
    """

    # open tfrecord file
    writer = tf.python_io.TFRecordWriter(tfrecord_file)

    # number of records
    m = len(labels)

    # loop over every example and write to tfrecord
    for i in range(m):
        example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(labels[i]),
                    'shape': _bytes_feature(shapes[i]),
                    'image': _bytes_feature(binary_imgs[i])
                    }))
        writer.write(example.SerializeToString())

    # close tfrecord
    writer.close()


def combine_files_with_labels(files, label):
    """
    Make a list of tuples. First entry of tuple being label and second being file name
    @params
        files: list of filenames
        labels: list of corresponding labels
    @returns
        file_w_label: List of Tuples of label and filenames
    """
    files_with_labels = []
    for file in files:
        files_with_labels.append((label, file))
    return files_with_labels


def write_tfrecord(path, target_dir, other_dir, tfrecord_file, size=None):
    """
    Write tfrecord for given path.
    @params
        path: Path to dataset
        target_dir: Directory Containing the positive targets
        other_dir: Directory Containing the negative targets
        size: Resize the images to given size. Default being None -> Do not change size.
    """
    if path[-1] != '/':
        path += '/'

    # get all file names
    target_files = os.listdir(path + target_dir)
    other_files = os.listdir(path + other_dir)

    # combine all files along with labels
    all_files = (combine_files_with_labels(target_files, 1) +
                 combine_files_with_labels(other_files, 0))

    # random shuffling
    shuffle(all_files)

    # Get labels shapes and binary_imgs to write tfrecord
    labels = []
    shapes = []
    binary_imgs = []

    for file_w_label in all_files:
        label = file_w_label[0]
        if label == 0:
            file_path = path + other_dir + '/' + file_w_label[1]
        else:
            file_path = path + target_dir + '/' + file_w_label[1]
        shape, img = get_binary_image(file_path, size=size)
        if shape == None:
            continue
        labels.append(label)
        shapes.append(shape)
        binary_imgs.append(img)

    # write to tfrecord file
    write_to_tfrecord(labels, shapes, binary_imgs, tfrecord_file)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                features={
                    'label': tf.FixedLenFeature([], tf.int64),
                    'shape': tf.FixedLenFeature([], tf.string),
                    'image': tf.FixedLenFeature([], tf.string)
                })

    shape = tf.decode_raw(features['shape'], tf.int32)
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape(Config.IMAGE_PIXELS)

    image = tf.cast(image, tf.float32) / 255 - 0.5
    label = tf.cast(features['label'], tf.float32)

    image = tf.reshape(image, [-1] + Config.IMAGE_SHAPE)
    label = tf.reshape(label, [-1, 1])

    return image, label


def run():
    write_tfrecord(Config.TRAIN_PATH, Config.TARGET_DIR, Config.OTHER_DIR,
                   Config.TFRECORD_TRAIN, Config.IMAGE_SIZE)
    write_tfrecord(Config.DEV_PATH, Config.TARGET_DIR, Config.OTHER_DIR,
                   Config.TFRECORD_DEV, Config.IMAGE_SIZE)
    write_tfrecord(Config.TEST_PATH, Config.TARGET_DIR, Config.OTHER_DIR,
                   Config.TFRECORD_TEST, Config.IMAGE_SIZE)


if __name__ == '__main__':
    run()