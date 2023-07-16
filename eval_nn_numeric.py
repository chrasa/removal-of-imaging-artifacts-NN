import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gc
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nn.losses import *
from nn.image_loader import load_images
from cmcrameri import cm


def print_table_head():
    header="""
\\begin{table}[H]
\\begin{center}
\\begin{tabular}{ | c | c | c | c | c | c | c | c |} 
\\hline
\\thead{Imaging \\\\ method} & 
\\thead{Model} & 
\\thead{Stride} & 
\\thead{Loss \\\\ function} &
\\thead{EMD} & 
\\thead{MSE} & 
\\thead{SSIM} & 
\\thead{Sobel} \\\\
\\hline"""
    print(header)


def print_table_footer(model, stride):
    footer=f"""
\\end{{tabular}}
\\end{{center}}
\\caption{{Resulting average evaluation metrics for {model} with stride {stride} implemented network architecture}}
\\label{{fig:sec12:metrics_{model}_{stride}}}
\\end{{table}}"""
    print(footer)


def evaluate_nn(imaging_method, model_name, loss_name, stride, n_images):
    if (stride == 2):
        resize = True
    else:
        resize = False

    _, _, x_test, y_test = load_images(imaging_method, n_images, 1.0, resize, "./training_data/training_data_normalized.npy")
    
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

    print(f"{imaging_method.upper()} & ")
    print(f"{model_name} & ")
    print(f"{stride} & ")
    print(f"{loss_name} & ")
    print(f"{np.mean(emds):.4f} & ")
    print(f"{np.mean(mses):.4f} & ")
    print(f"{np.mean(ssims):.4f} & ")
    print(f"{np.mean(sobel):.4f} \\\\ ")
    print(f"\\hline")


# def eval_model(model, loss, stride, n_images):
#     print_table_head()
#     eval_rtm_rom(model, loss, stride, n_images)
#     print_table_footer(model, stride)


def eval_rtm_rom(model, loss, stride, n_images):
    evaluate_nn('rtm', model, loss, stride, n_images)
    evaluate_nn('rom', model, loss, stride, n_images)


if __name__ == "__main__":
    n_images = int(sys.argv[1])

    print_table_head()

    eval_rtm_rom('ConvAuto', 'sobel', 2, n_images)

    eval_rtm_rom('ConvAuto', 'sobel', 5, n_images)
    
    eval_rtm_rom('ConvAuto', 'ssim', 2, n_images)
    
    eval_rtm_rom('ConvAuto', 'ssim', 5, n_images)

    eval_rtm_rom('ConvNN', 'sobel', 5, n_images)
    
    eval_rtm_rom('ConvNN', 'ssim', 5, n_images)

    eval_rtm_rom('ResNet', 'sobel', 2, n_images)
    
    eval_rtm_rom('ResNet', 'sobel', 5, n_images)
    
    eval_rtm_rom('ResNet', 'ssim', 2, n_images)
    
    eval_rtm_rom('ResNet', 'ssim', 5, n_images)

    eval_rtm_rom('UNet', 'sobel', 2, n_images)
    
    eval_rtm_rom('UNet', 'sobel', 5, n_images)
    
    eval_rtm_rom('UNet', 'ssim', 2, n_images)
    
    eval_rtm_rom('UNet', 'ssim', 5, n_images)

    print_table_footer('model', 'stride')