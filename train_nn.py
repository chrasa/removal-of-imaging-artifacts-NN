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



def train_model(x_train, y_train, model_name, loss_name, stride):
    artifact_remover = NetworkGenerator.get_model(model_name, stride)

    # loss, early stopping and optimizer
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = sobel_loss if loss_name == "sobel" else ssim_loss
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=50,
                                                      restore_best_weights=True)

    artifact_remover.compile(loss=loss, optimizer=optim)
    artifact_remover.fit(x_train,
            y_train,
            epochs=4,
            shuffle=False,
            batch_size=10,
            verbose=2,
            callbacks=[early_stopping],
            validation_data=(x_test, y_test))

    artifact_remover.save(f"./saved_model/{model_name}_{loss_name}_{stride}_trained_model.h5")
    return artifact_remover

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
    
    print(f"\nTraining model {model_name} with loss {loss_name}...\n")
    artifact_remover = train_model(x_train, y_train, model_name, loss_name, stride)

