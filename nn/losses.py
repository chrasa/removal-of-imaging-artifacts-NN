import numpy as np
import tensorflow as tf
from scipy.stats import wasserstein_distance;

def calculate_emd(target, predicted):

    ws_distances = []
    for i in range(target.shape[0]):
        t_hist, _ = np.histogram(target[i, :, :, :], bins=256, density = True)
        p_hist, _ = np.histogram(predicted[i, :, :, :], bins=256, density = True)

        ws_distances.append(wasserstein_distance(t_hist, p_hist))
    return np.mean(ws_distances)


def calculate_mse(target, predicted):
    mse = tf.keras.losses.MeanSquaredError()
    
    return mse(target, predicted)


def calculate_ssim(target, predicted):

    return tf.reduce_mean(tf.image.ssim(tf.cast(target, tf.float64), tf.cast(predicted, tf.float64), max_val=1.0))

def calculate_sobel(target, predicted):
    return sobel_loss(tf.cast(target, tf.float64), tf.cast(predicted, tf.float64))
    

def sobel_loss(target, predicted):
    sobel_target = tf.image.sobel_edges(target)
    sobel_predicted = tf.image.sobel_edges(predicted)
    
    return tf.reduce_mean(tf.square(tf.math.subtract(sobel_target, sobel_predicted)))


def ssim_loss(target, predicted):
    loss = tf.reduce_mean(1 - tf.image.ssim(target, predicted, max_val=1.0))
    return loss