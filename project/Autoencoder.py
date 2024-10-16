import mesa_web as mw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import re

from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import layers, losses
from keras.models import Model
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

##Autoencoder class definition
class Autoencoder(Model):
    def __init__(self, encoder_neurons, decoder_neurons, encoder_activations, decoder_activations):
        ##input check
        if not len(encoder_neurons) == len(encoder_activations):
            raise ValueError('The vector of neuron numbers for the encoder should be the same size of the activations')
        if not len(decoder_neurons) == len(decoder_activations):
            raise ValueError('The vector of neuron numbers for the decoder should be the same size of the activations')

        #encoder_layers = []
        #decoder_layers = []

        ##define the encoder
        input_shape = keras.Input(shape=(decoder_neurons[-1],))
        print(input_shape)
        print(type(input_shape))
        encoded = layers.Dense(encoder_neurons[0], activation=encoder_activations[0])(input_shape)
        for i in range(1, len(encoder_neurons)):
            encoded = layers.Dense(encoder_neurons[i], activation=encoder_activations[i])(encoded)

        ##define the decoder
        decoded = layers.Dense(decoder_neurons[0], activation=decoder_activations[0])(encoded)
        for i in range(1, len(decoder_neurons)):
            decoded = layers.Dense(decoder_neurons[i], activation=decoder_activations[i])(decoded)

        super().__init__(input_shape, decoded)

