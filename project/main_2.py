import mesa_web as mw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 
import re

from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras import layers, losses, metrics
from keras.datasets import fashion_mnist
from keras.models import Model
from scipy.interpolate import UnivariateSpline

## Autoencoder class definition
class Autoencoder(Model):
    
    def __init__(self, encoder_neurons, decoder_neurons, encoder_activations, decoder_activations):
        super(Autoencoder, self).__init__()
        
        # Input validation
        if len(encoder_neurons) != len(encoder_activations):
            raise ValueError('The vector of neuron numbers for the encoder should be the same size as the activations')
        if len(decoder_neurons) != len(decoder_activations):
            raise ValueError('The vector of neuron numbers for the decoder should be the same size as the activations')

        self.encoder_layers = []
        self.decoder_layers = []

        """
        # Define the encoder
        self.encoder_layers.append(layers.InputLayer(input_shape=(decoder_neurons[-1],)))
        for neurons, activation in zip(encoder_neurons, encoder_activations):
            self.encoder_layers.append(layers.Dense(neurons, activation=activation))

        # Define the decoder
        for neurons, activation in zip(decoder_neurons, decoder_activations):
            self.decoder_layers.append(layers.Dense(neurons, activation=activation))

        self.encoder = tf.keras.Sequential(self.encoder_layers)
        self.decoder = tf.keras.Sequential(self.decoder_layers)
        """
        self.encoder = tf.keras.Sequential([layers.InputLayer(shape=(50,)),layers.Dense(100,'relu'),layers.Dense(2,'relu')])
        self.decoder = tf.keras.Sequential([layers.Dense(100,'relu'),layers.Dense(50,'sigmoid')])


    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Import parameters
cwd = os.getcwd()
dir_names = ['MESA-Web_M07_Z00001', 'MESA-Web_M07_Z002', 'MESA-Web_M10_Z002', 'MESA-Web_M10_Z0001']
column_filter = ['mass', 'radius', 'initial_mass', 'initial_z', 'star_age', 'logRho', 'logT', 'Teff', 'energy', 'photosphere_L', 'photosphere_r', 'star_mass', 'h1', 'he3', 'he4']
column_filter_train = ['mass', 'radius', 'logRho', 'logT', 'energy', 'h1', 'he3', 'he4']
n_points = 50
r = np.linspace(0, 1, n_points)

for i, dir_name in enumerate(dir_names):
    print(f"####\t\tIMPORTING DATA FROM FOLDER {dir_name}\t\t####")
    dir_name = os.path.join(cwd, 'StellarTracks', dir_name)

    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')

    filenames = [filename for filename in os.listdir(dir_name) if re.fullmatch(r'profile[0-9]+\.data', filename)]
    filenames = sorted(filenames, key=extract_number)

    for j, filename in enumerate(filenames):
        print(f"####\t\t\tIMPORTING FILE {filename}\t\t\t####")
        filename = os.path.join(dir_name, filename)

        data = mw.read_profile(filename)  # Assuming mw.read_profile is defined elsewhere

        profile_df = pd.DataFrame(data)
        filtered_profile_df = profile_df[column_filter].copy()
        train_filtered_profile_df = profile_df[column_filter_train].copy()

        # Normalization process
        tot_radius = filtered_profile_df['photosphere_r']

        norm_radius = (filtered_profile_df['radius'] - filtered_profile_df['radius'].min()) / (tot_radius - filtered_profile_df['radius'].min())
        norm_radius = np.asarray(norm_radius.T)
        log_rho = np.asarray(train_filtered_profile_df['logRho'].T)
        int_log_rho = UnivariateSpline(norm_radius, log_rho, k=2, s=0)(r)

        train_df = pd.DataFrame(data=int_log_rho.T, columns=[f"log_rho_{i}_{j}"])

        if i == 0 and j == 0:
            linear_train_df = train_df
        else:
            linear_train_df = pd.concat([linear_train_df, train_df], axis=1)

print(linear_train_df.shape)
print(linear_train_df)

x_train, x_test = train_test_split(linear_train_df.T, test_size=0.2)
x_train = x_train.values
x_test = x_test.values
print(x_train.shape)
print(x_test.shape)
shape = x_test.shape[1:]
save_graphs = True

for i in range(1, 10):
    encoder_neurons = [100, i]
    decoder_neurons = encoder_neurons[:-1][::-1]
    decoder_neurons.append(n_points)
    encoder_activations = ['relu'] * len(encoder_neurons)
    decoder_activations = ['relu'] * (len(decoder_neurons) - 1)
    decoder_activations.append('sigmoid')

    autoencoder = Autoencoder(encoder_neurons=encoder_neurons, decoder_neurons=decoder_neurons,
                              encoder_activations=encoder_activations, decoder_activations=decoder_activations)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.summary()
    
    history = autoencoder.fit(x_train, x_train,
                              epochs=150,
                              shuffle=True,
                              validation_data=(x_test, x_test))

    if save_graphs:
        file_save_dir = os.path.join(os.getcwd(), "Graphs", f"TrainValLoss_dim_{i}.png")
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title(f'Training Loss VS Validation Loss - Latent Dim = {i}')
        plt.legend()
        plt.savefig(file_save_dir)
        plt.close()

    decoded_imgs = autoencoder.predict(x_test)
    print(decoded_imgs.shape)

    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    plt.title(f'Examples of fit with latent dimension = {i}')
    for j in range(n):
        # Display original
        ax = plt.subplot(2, n, j + 1)
        plt.scatter(r, x_test[j])
        plt.gray()

        # Display reconstruction
        ax = plt.subplot(2, n, j + 1 + n)
        plt.scatter(r, decoded_imgs[j])
        plt.gray()
    plt.show()
