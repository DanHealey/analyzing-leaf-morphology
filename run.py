import os
import argparse

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

    img_width, img_height = 256, 256
    data_dir = './processed_images'
    epochs = 50
    batch_size = 16

    #Choose model
    if ARGS.model == "ours":
        model = our_model(img_width, img_height)
    else:
        model = paper_model(img_width, img_height)

    # Get image data 
    train_datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.05,
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        data_dir, 
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation')

    #Fit model
    model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // batch_size,
        epochs = epochs)

    model.summary()

# Make arguments global
ARGS = parse_args()

main()
