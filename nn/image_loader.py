import gc
import numpy as np
import tensorflow as tf
from setup import ImageSetup


def load_images(imaging_method: str, n_images: int, validation_split: float, resize: bool):
    print("Loading training data from disc...")
    setup = ImageSetup()

    training_data = np.load("./training_data/training_data_normalized.npy")
    print(f"Shape of training_data: {training_data.shape}")
    training_data = training_data[:n_images,:,:]
    gc.collect()
    # print(training_data.shape)
    # training_data = np.reshape(training_data, (n_images, 3, setup.N_x_im, setup.N_y_im))
    # x_image_tensor = training_data[:,1,:]
    # x_image_tensor = x_image_tensor[:,:,:175]

    # y_image_tensor = training_data[:,0,:]
    # y_image_tensor = x_image_tensor[:,:,:175]

    if imaging_method.lower() == 'rtm':
        imaging_method_idx = 1
    elif imaging_method.lower() == 'rom':
        imaging_method_idx = 2
    else:
        raise ValueError(f"The imaging_method '{imaging_method}' is not a valid method")

    x_img_array_list = []
    y_img_array_list = []

    for i in range(n_images):
        x_img_array = training_data[i,imaging_method_idx,:]
        x_img_array = x_img_array.reshape(350,180)
        x_img_array = x_img_array[:,:175]
        x_img_array_list.append(preprocess_data(x_img_array))

        y_img_array = training_data[i,0,:]
        y_img_array = y_img_array.reshape(350,180)
        y_img_array = y_img_array[:,:175]
        y_img_array_list.append(preprocess_data(y_img_array))


    x_image_tensor = np.stack(x_img_array_list, axis=0)
    y_image_tensor = np.stack(y_img_array_list, axis=0)

    training_index = int(x_image_tensor.shape[0]*(1-validation_split))
    # x_train_images = training_data[:training_index, 0, :, :]
    # y_train_images = training_data[:training_index, 1, :, :]
    x_train_images = x_image_tensor[:training_index, :, :]
    y_train_images = y_image_tensor[:training_index, :, :]

    x_test_images = x_image_tensor[training_index:, :, :]
    y_test_images = y_image_tensor[training_index:, :, :]

    x_train_images = x_train_images[..., tf.newaxis]
    y_train_images = y_train_images[..., tf.newaxis]
    x_test_images = x_test_images[..., tf.newaxis]
    y_test_images = y_test_images[..., tf.newaxis]

    if resize:
        x_train_images = tf.image.resize(x_train_images, (344, 168))
        y_train_images = tf.image.resize(y_train_images, (344, 168))
        x_test_images = tf.image.resize(x_test_images, (344, 168))
        y_test_images = tf.image.resize(y_test_images, (344, 168))

    return x_train_images, y_train_images, x_test_images, y_test_images


def preprocess_data(image_array: np.array):
    return (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))