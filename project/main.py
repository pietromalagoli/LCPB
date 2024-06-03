import mesa_web as mw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import re
import aux_functions as aux

from Autoencoder import Autoencoder
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import layers, losses
from keras.models import Model
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

##########IMPORT PARAMETERS##########
cwd = os.getcwd()
dir_names = ['all']  # type 'all' if you want to use all the data
column_filter = ['mass', 'radius', 'initial_mass', 'initial_z', 'star_age', 'logRho', 'logT',
                 'Teff', 'energy', 'photosphere_L', 'photosphere_r', 'star_mass', 'h1', 'he3', 'he4']
column_filter_train = ['logRho','logT']  # radius is not included for coding reasons but is still considered
n_points = 100  # n of points to sample from each profile
r = np.linspace(0, 1, n_points + 1)[1:]  # values of normalized r on which to take the values of the variables

if dir_names[0] == 'all':
    dir_names = ['MESA-Web_M07_Z00001', 'MESA-Web_M07_Z002', 'MESA-Web_M10_Z002', 'MESA-Web_M10_Z0001',
                 'MESA-Web_M10_Z00001', 'MESA-Web_M15_Z0001', 'MESA-Web_M15_Z00001', 'MESA-Web_M30_Z00001',
                 'MESA-Web_M30_Z002', 'MESA-Web_M50_Z00001', 'MESA-Web_M50_Z002', 'MESA-Web_M50_Z001',
                 'MESA-Web_M5_Z002', 'MESA-Web_M5_Z0001', 'MESA-Web_M1_Z00001', 'MESA-Web_M1_Z0001']
    
##HYPERPARAMETERS
encoder_neurons_list=[[100],
                      [200,100],
                      [500,250,100],
                      [500,250,100,50]]    #without hidden dimension
activations=['leaky_relu']    #same for each layer
optimizers=['adam','rmsprop']
losses=[losses.MeanSquaredError()]

##CODE
all_profiles=aux.get_data(dir_names=dir_names,column_filter=column_filter,\
                          column_filter_train=column_filter_train,r=r)

for encoder_neurons in encoder_neurons_list:
    for activation in activations:
        for optimizer in optimizers:
            for loss in losses:
                string_neurons = ""
                for i in encoder_neurons:
                    string_neurons += f"{i}_"
                folder = os.path.join("Graphs", f"{string_neurons}{activation}_{optimizer}_{loss.name}")
                model = aux.train_autoencoder(
                    all_profiles=all_profiles,
                    encoder_neurons_in=encoder_neurons,
                    activation=activation,
                    optimizer=optimizer,
                    plot_loss=True,
                    plot_reconstructed=True,
                    save_model=True,
                    folder=folder,
                    column_filter_train=column_filter_train,
                    loss=loss
                )
