import numpy as np
import keras
import skimage
import os

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, data_location, batch_size=32, dim=(500, 500),
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_location = data_location

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_for_batch = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_for_batch)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_for_batch):
        'Generates data containing batch_size samples'
        X = np.zeros((self.batch_size, *self.dim))
        y = np.zeros((self.batch_size))

        count = 0
        for root, dirs, files in os.walk(self.data_location, topdown=False):
            for name in files:
                if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
                    try:
                        X[count, ] = skimage.io.imread(os.path.join(root, name))
                        y[count] = 1
                    except Exception as e:
                            print(e)
        return X, y