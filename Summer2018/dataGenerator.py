import numpy as np
import keras
import scipy.misc as misc
from random import shuffle, randint

class DataGenerator(keras.utils.Sequence):
    """
    DataGenerator for keras training
    """
    def __init__(self, list_IDs, labels=None, batch_size=20, dim=(500,500), n_channels=1,
                 n_classes=124, shuffle=True, crop=True, crop_size=(500,500), load_npy=True):
        """
        Initialization
        """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.crop = crop
        self.crop_size = crop_size
        self.load_npy = load_npy
        self.on_epoch_end()

    def __len__(self):
        """
        Return the number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Update indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generate data containing batch_size samples
        # X : (n_samples, *dim, n_channels)
        # y : (n_samples, *dim, n_channels)
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            if self.load_npy:
                img = np.load(ID)
                annotation = np.load(ID.replace("/images_png/", "/pix_annotations_png/"))
            else:
                img = misc.imread(ID, flatten=True)
                annotation = misc.imread(ID.replace("/images_png/", "/pix_annotations_png/"))

            img = np.expand_dims(img, axis=-1)
            annotation = np.expand_dims(annotation, axis=-1)

            coord_0 = randint(0, (img.shape[0] - self.crop_size[0]))
            coord_1 = randint(0, (img.shape[1] - self.crop_size[1]))
            img = img[coord_0:(coord_0+self.crop_size[0]), coord_1:(coord_1+self.crop_size[1])]
            annotation = annotation[coord_0:(coord_0+self.crop_size[0]), coord_1:(coord_1+self.crop_size[1])]

            X[i,:] = img
            y[i,:] = annotation

        return X, y
