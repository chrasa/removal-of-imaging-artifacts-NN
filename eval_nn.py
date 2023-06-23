import gc
import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import convolve2d as conv2
import sys, getopt
# from PIL import Image
from setup import ImageSetup
from nn.networks import NetworkGenerator
from nn.losses import *
from nn.image_loader import load_images

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tf.config.run_functions_eagerly(True)



def plot_comparison(n_images,
                    imaging_result,
                    reconstructed_images,
                    label_images,
                    model_name,
                    loss_name,
                    stride,
                    start_index):
    
    save_path = f"images/pngs/{model_name}_{loss_name}_{stride}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(n_images):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.gray()

        plot_image(ax1, imaging_result[i], "Imaging algorithm result")

        plot_image(ax2, reconstructed_images[i], "Output of CNN")

        plot_image(ax3, label_images[i], "Actual fracture image")

        plt.savefig(f"{save_path}/im{i+start_index}")

        fig, ax = plt.subplots(1, 1)
        plot_image(ax, reconstructed_images[i], f"Output of {model_name}")
        plt.savefig(f"{save_path}/out{i+start_index}")


    print("Images saved.")


def plot_image(ax, image, title):
    ax.imshow(tf.squeeze(image))
    ax.set_title(title)
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 


if __name__ == "__main__":

    if not os.path.exists("./images"):
        os.makedirs("./images/data")
        os.makedirs("./images/labels")
        os.makedirs("./images/pngs")

    if not os.path.exists("./saved_model/"):
        os.makedirs("./saved_model/")

    stride = int(sys.argv[3])

    resize = False
    if (stride == 2):
        resize = True

    model_name = sys.argv[1]
    loss_name = sys.argv[2]
    n_images = int(sys.argv[5])

    x_train, y_train, x_test, y_test = load_images(n_images, 0.2, resize)
    
    print(f"\nLoading model {model_name} with loss {loss_name}...\n")
    artifact_remover = tf.keras.models.load_model(f"./saved_model/{model_name}_{loss_name}_{stride}_trained_model.h5", compile=False)

    # set loss and optimizer here
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = sobel_loss if loss_name == "sobel" else ssim_loss
    metrics = ["mse"]
    artifact_remover.compile(metrics=metrics, loss=loss, optimizer=optim)

    #artifact_remover.evaluate(x_test, y_test)

    # one_frac_x, one_frac_y = get_images("one_frac.npy", resize)
    # point_scatter_x, point_scatter_y = get_images("point_scatter.npy", resize)
    # ten_frac_x, ten_frac_y = get_images("ten_frac.npy", resize)
    # two_close_x, two_close_y = get_images("two_close.npy", resize)

    # x_special = np.stack([one_frac_x, point_scatter_x, ten_frac_x, two_close_x], axis=0)
    # y_special = np.stack([one_frac_y, point_scatter_y, ten_frac_y, two_close_y], axis=0)

    emds = []
    mses = []
    ssims = []
    sobel = []

    im_per_eval = 20

    for i in range(6):
        current_decoded_images = artifact_remover(x_test[i*im_per_eval:im_per_eval*i + im_per_eval+1])
        emds.append(calculate_emd(y_test[i*im_per_eval:im_per_eval*i + im_per_eval+1], current_decoded_images))
        mses.append(calculate_mse(y_test[i*im_per_eval:im_per_eval*i + im_per_eval+1], current_decoded_images))
        ssims.append(calculate_ssim(y_test[i*im_per_eval:im_per_eval*i + im_per_eval+1], current_decoded_images))
        sobel.append(calculate_sobel(y_test[i*im_per_eval:im_per_eval*i + im_per_eval+1], current_decoded_images))

    # special_images = artifact_remover(x_special)
    decoded_images = artifact_remover(x_test[:im_per_eval+1])

    print("Average earth mover distance: ", np.mean(emds))
    print("Average mean squared error: ", np.mean(mses))
    print("Average SSIM: ", np.mean(ssims))
    print("Average sobel loss: ", np.mean(sobel))

    # plot_comparison(4, x_special, special_images, y_special, model_name, loss_name, stride, 0)
    # plot_comparison(im_per_eval, x_test[:im_per_eval+1], decoded_images, y_test[:im_per_eval+1], model_name, loss_name, stride, 4)