"""
We need to prepare the data for feeding model.
We have these functions to preprocess our data:

1 -> tensorizing the images
2 -> one hot encoding the classes

Then, our data will be prepared.

"""

import glob
import os
import cv2
import torch
import numpy as np
from constant import *
import tqdm

def tensorize_image(image_path_list:list,
                    output_shape: tuple,
                    cuda=False):
    """
    :param image_path_list: list of string
    The path of images that will be found
    :param output_shape: tuple of integers
    (d1, d2): d1 and d2 are width and height of the DNN model's input.
    :param cuda: boolean
    For cuda support
    :return: torch tensor
    """
    # Create empty list
    local_image_list = []

    # For each image
    for image_path in image_path_list:
        # Access and read image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Unable to read image at {image_path}. Skipping this image.")
            continue

        # Resize the image according to defined shape
        image = cv2.resize(image, output_shape)

        # Change input structure according to pytorch input structure
        torchlike_image = torchlike_data(image)

        # Add into the list
        local_image_list.append(torchlike_image)

    # Convert from list structure to torch tensor
    image_array = np.array(local_image_list, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()

    # If multiprocessing is chosen
    if cuda:
        torch_image = torch_image.cuda()

    return torch_image

def torchlike_data(data):
    """
        Change data structure according to Torch Tensor structure where the first
        dimension corresponds to the data depth.


        Parameters
        ----------
        data : Array of uint8
            Shape : HxWxC.

        Returns
        -------
        torchlike_data_output : Array of float64
            Shape : CxHxW.

        """

    # Obtain channel value of the input
    n_channels = data.shape[2]

    # Create and empty image whose dimension is similar to input
    torchlike_data_output = np.empty((n_channels, data.shape[0], data.shape[1]))

    # For each channel
    """
    for c in range(n_channels):
        for h in range(data.shape[0]):
            for w in range(data.shape[1]):
                torchlike_data_output[c][h][w] = data[h][w][c]
    """
    for c in range(n_channels):
        torchlike_data_output[c] = data[:, :, c]

    return torchlike_data_output


def one_hot_encoder(dim1_array, numClasses, cuda):
    """
    Changes data to understandable for model. If data stays as numerical, then model can estimate tomato_healthy(20) is bigger than
    apple_healthy(0). We have to change it something different but not numerical.
    For example -> [1, 0, 0] = 1, [0, 1, 0] = 2, [0, 0, 1] = 3
    We will take dim1_array which is numerical and change it to the format.
    :param dim1_array: list of int, numerical category of each plant_disease
    :param numClasses: int, how many plant_disease we have
    :param cuda: bool, for gpu support
    :return: torch.Tensor
    """
    empty = torch.zeros((dim1_array.shape[0], numClasses))

    for i in range(dim1_array.shape[0]):
        idx = dim1_array[i].item()
        for k in range(numClasses):
            empty[i, k][k == idx] = 1
    return empty.cuda() if cuda else empty


if __name__ == '__main__':

    image = glob.glob(os.path.join(TRAIN_DIR + '\\*', '*'))
    print("len of image", len(image))

    tensorizedImage = tensorize_image(image, (224, 224), True)

    print("For features:\ndtype is " + str(tensorizedImage.dtype))
    print("Type is " + str(type(tensorizedImage)))
    print("The size should be [" + str(4) + ", 3, " + str(224) + ", " + str(224) + "]")
    print("Size is " + str(tensorizedImage.shape) + "\n")








