import os
import argparse

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import datetime

from models import our_model, paper_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        default='ours',
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
        default='.'+os.sep+'epidermal_data_model'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--batch_size',
        default=16,
        help='batch size, integer')
    parser.add_argument(
        '--epochs',
        default=50,
        help='batch size, integer')

    return parser.parse_args()


def main():
    """ Main function. """

    train_data_dir = ARGS.data
    test_data_dir = "./val/"

    img_width, img_height = 256, 256
    data_dir = 'C:/Users/henrycs/Documents/GitHub/analyzing-leaf-morphology/processed_images'
    epochs = int(ARGS.epochs)
    batch_size = int(ARGS.batch_size)

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_bs_{b}_epochs_{e}".format(b=batch_size,e=epochs)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,write_graph=False)

    #Choose model
    if ARGS.model == "ours":
        model = our_model(img_width, img_height)
    else:
        model = paper_model(img_width, img_height)

    # Get image data 
    train_datagen = ImageDataGenerator(
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2)

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='binary',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir, 
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='binary',
        subset='validation')

    testing_generator = test_datagen.flow_from_directory(
        test_data_dir, 
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='binary')

    '''
    #Saves best model weights from training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'model_weights.h5', 
        monitor='loss', 
        verbose=1, 
        save_best_only=True, 
        mode='min',
        period=1)
    '''

    #Fit model
    model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // batch_size,
        epochs = epochs, 
        callbacks=[tensorboard_callback])

        
    model.save_weights("model_weights.h5")
    model.evaluate(
        testing_generator
    )
    # model.compute_metrics()
    # model.get_metrics_results()

# Make arguments global
ARGS = parse_args()

main()
