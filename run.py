import os
import argparse

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from models import our_model, paper_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        default='paper',
        choices=['ours', 'paper'],
        help='''Which model to run -
        our model (ours), or the model used in the paper (paper).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--data',
        default='.'+os.sep+'labelled_data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    return parser.parse_args()


def main():
    """ Main function. """

    img_width, img_height = 500, 500
    data_dir = './labelled_data'
    epochs = 50
    batch_size = 16

    #Choose model
    if ARGS.model == "ours":
        model = our_model(img_width, img_height)
    else:
        model = paper_model(img_width, img_height)

    # Get image data (does not work for .tif images so we would have to convert to jpg/png/etc or use ImageDataGenerator)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        color_mode="grayscale",
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        color_mode="grayscale",
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    #Fit model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    model.summary()

# Make arguments global
ARGS = parse_args()

main()
