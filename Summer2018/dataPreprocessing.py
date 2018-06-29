import glob
import os
from random import shuffle

def splitTrainValidation(deepscores_path, max_size=1000, test_size=30, npy_only=True):
    """
    Return two non-overlapped lists of images randomly shuffled.
    """
    images_list = []
    image_paths = None
    if npy_only:
        image_paths = os.path.join(deepscores_path, 'images_png', '*.npy')
    else:
        image_paths = os.path.join(deepscores_path, 'images_png', '*.png')
    images_list.extend(glob.glob(image_paths))
    
    assert test_size < max_size, "#{Test images} have to be smaller than #{All images}"
    assert max_size <= len(images_list), "Number of images isn't enough"

    test_image_list = images_list[0:test_size]
    train_image_list = images_list[test_size:max_size]

    return train_image_list, test_image_list
    