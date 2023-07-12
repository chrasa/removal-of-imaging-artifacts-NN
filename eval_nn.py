import gc
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nn.losses import *
from nn.image_loader import load_images

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tf.config.run_functions_eagerly(True)



def plot_comparison(n_images,
                    imaging_output,
                    cnn_output,
                    fracture_image,
                    imaging_method,
                    model_name,
                    loss_name,
                    stride):
    
    save_path = f"images/pngs/{imaging_method}_{model_name}_{loss_name}_{stride}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(n_images):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.gray()

        plot_image(ax1, imaging_output[i], "RTM output")
        plot_image(ax2, cnn_output[i], "CNN output")
        plot_image(ax3, fracture_image[i], "Fracture image")

        plt.savefig(f"{save_path}/im{i}")
        plt.close()

        fig, ax = plt.subplots(1, 1)
        plot_image(ax, cnn_output[i], f"Output of {model_name}")
        plt.savefig(f"{save_path}/out{i}")
        plt.close()

    print("Images saved.")


def plot_image(ax, image, title):
    ax.imshow(tf.squeeze(image))
    ax.set_title(title)
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 


def evaluate_nn(imaging_method, model_name, loss_name, stride, n_images):
    if (stride == 2):
        resize = True
    else:
        resize = False

    _, _, x_test, y_test = load_images('rtm', n_images, 0.2, resize)
    
    print(f"\nLoading model {model_name} with loss {loss_name}...\n")
    artifact_remover = tf.keras.models.load_model(f"./saved_model/{imaging_method}_{model_name}_{loss_name}_{stride}_trained_model.h5", compile=False)

    # set loss and optimizer here
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = sobel_loss if loss_name == "sobel" else ssim_loss
    metrics = ["mse"]
    artifact_remover.compile(metrics=metrics, loss=loss, optimizer=optim)

    emds = []
    mses = []
    ssims = []
    sobel = []

    decoded_images = artifact_remover(x_test)
    emds.append(calculate_emd(y_test, decoded_images))
    mses.append(calculate_mse(y_test, decoded_images))
    ssims.append(calculate_ssim(y_test, decoded_images))
    sobel.append(calculate_sobel(y_test, decoded_images))

    print("Average earth mover distance: ", np.mean(emds))
    print("Average mean squared error: ", np.mean(mses))
    print("Average SSIM: ", np.mean(ssims))
    print("Average sobel loss: ", np.mean(sobel))

    plot_comparison(int(0.2*n_images), x_test, decoded_images, y_test, imaging_method, model_name, loss_name, stride)

if __name__ == "__main__":
    n_images = int(sys.argv[1])

    ### RTM ###
    evaluate_nn('rtm', 'ConvAuto', 'sobel', 2, n_images)
    evaluate_nn('rtm', 'ConvAuto', 'sobel', 5, n_images)
    evaluate_nn('rtm', 'ConvAuto', 'ssim', 2, n_images)
    evaluate_nn('rtm', 'ConvAuto', 'ssim', 5, n_images)

    evaluate_nn('rtm', 'ConvNN', 'sobel', 5, n_images)
    evaluate_nn('rtm', 'ConvNN', 'ssim', 5, n_images)

    evaluate_nn('rtm', 'ResNet', 'sobel', 2, n_images)
    evaluate_nn('rtm', 'ResNet', 'sobel', 5, n_images)
    evaluate_nn('rtm', 'ResNet', 'ssim', 2, n_images)
    evaluate_nn('rtm', 'ResNet', 'ssim', 5, n_images)

    # evaluate_nn('rtm', 'UNet', 'sobel', 2, n_images)
    # evaluate_nn('rtm', 'UNet', 'sobel', 5, n_images)
    # evaluate_nn('rtm', 'UNet', 'ssim', 2, n_images)
    # evaluate_nn('rtm', 'UNet', 'ssim', 5, n_images)

    ### ROM ###
    evaluate_nn('rom', 'ConvAuto', 'sobel', 2, n_images)
    evaluate_nn('rom', 'ConvAuto', 'sobel', 5, n_images)
    evaluate_nn('rom', 'ConvAuto', 'ssim', 2, n_images)
    evaluate_nn('rom', 'ConvAuto', 'ssim', 5, n_images)

    evaluate_nn('rom', 'ConvNN', 'sobel', 5, n_images)
    evaluate_nn('rom', 'ConvNN', 'ssim', 5, n_images)

    evaluate_nn('rom', 'ResNet', 'sobel', 2, n_images)
    evaluate_nn('rom', 'ResNet', 'sobel', 5, n_images)
    evaluate_nn('rom', 'ResNet', 'ssim', 2, n_images)
    evaluate_nn('rom', 'ResNet', 'ssim', 5, n_images)

    evaluate_nn('rom', 'UNet', 'sobel', 2, n_images)
    evaluate_nn('rom', 'UNet', 'sobel', 5, n_images)
    evaluate_nn('rom', 'UNet', 'ssim', 2, n_images)
    evaluate_nn('rom', 'UNet', 'ssim', 5, n_images)
