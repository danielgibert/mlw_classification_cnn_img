import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import math
import PIL

def parse_labels(labels_batch, num_classes):
    """
    Convert integers to one-hot vectors

    Parameters
    ----------
    labels_batch: list
        Batch of labels
    num_classes: int
        Number of classes/families

    Return
    ------
    y_batch: list of one-hot vectors
    """
    y_batch = []
    for label in labels_batch:
        y = np.zeros(num_classes)
        y[label] = 1
        y_batch.append(y)
    return y_batch


def convert_mlw_to_img(self, bytes_filepath, width=28, height=28):
    """
    It represents the hexadecimal content as an image by interpreting every byte as the gray-level of one pixel in
    an image. The resulting images have very fine texture patterns and can be used to extract visual signatures
    for each malware family

    Parameters
    ----------
    bytes_filepath: str
        Filepath of the bytes file
    width: int
        Width of the resulting image
    height: int
        Height of the resulting image

    Return
    ------
    img: np.array
        Array-like structure containing the representation of the malware sample as a gray-scale image
    """

    with open(bytes_filepath) as hex_file:
        #Extract hex values
        hex_array = []
        for line in hex_file.readlines():
            hex_values = line.split()
            if len(hex_values) != 17:
                continue
            hex_array.append([int(i, 16) if i != '??' else 0 for i in hex_values[1:]])
        hex_array = np.array(hex_array)

    #Convert to an image of a specific size
    if hex_array.shape[1] != 16:
        assert (False)
    b = int((hex_array.shape[0] * 16) ** (0.5))
    b = 2 ** (int(math.log(b) / math.log(2)) + 1)
    a = int(hex_array.shape[0] * 16 / b)
    hex_array = hex_array[:a * b / 16, :]
    im = np.reshape(hex_array, (a, b))

    if width is not None and height is not None:
        #Downsample
        im = PIL.Image.fromarray(np.uint8(im))
        im = im.resize((width, height), resample=PIL.Image.ANTIALIAS)
        im = np.array(im.getdata())

    return im