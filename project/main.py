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

##########IMPORT PARAMETERS##########
cwd = os.getcwd()
dir_names = ['MESA-Web_M07_Z00001', 'MESA-Web_M10_Z002']  # type 'all' if you want to use all the data
column_filter = ['mass', 'radius', 'initial_mass', 'initial_z', 'star_age', 'logRho', 'logT',
                 'Teff', 'energy', 'photosphere_L', 'photosphere_r', 'star_mass', 'h1', 'he3', 'he4']
column_filter_train = ['logRho','mass']  # radius is not included for coding reasons but is still considered
n_points = 100  # n of points to sample from each profile
r = np.linspace(0, 1, n_points + 1)[1:]  # values of normalized r on which to take the values of the variables

if dir_names[0] == 'all':
    dir_names = ['MESA-Web_M07_Z00001', 'MESA-Web_M07_Z002', 'MESA-Web_M10_Z002', 'MESA-Web_M10_Z0001',
                 'MESA-Web_M10_Z00001', 'MESA-Web_M15_Z0001', 'MESA-Web_M15_Z00001', 'MESA-Web_M30_Z00001',
                 'MESA-Web_M30_Z002', 'MESA-Web_M50_Z00001', 'MESA-Web_M50_Z002', 'MESA-Web_M50_Z001',
                 'MESA-Web_M5_Z002', 'MESA-Web_M5_Z0001', 'MESA-Web_M1_Z00001', 'MESA-Web_M1_Z0001']


##Autoencoder class definition
class Autoencoder(Model):
    def __init__(self, encoder_neurons, decoder_neurons, encoder_activations, decoder_activations):
        ##input check
        if not len(encoder_neurons) == len(encoder_activations):
            raise ValueError('The vector of neuron numbers for the encoder should be the same size of the activations')
        if not len(decoder_neurons) == len(decoder_activations):
            raise ValueError('The vector of neuron numbers for the decoder should be the same size of the activations')

        encoder_layers = []
        decoder_layers = []

        ##define the encoder
        input_shape = keras.Input(shape=(decoder_neurons[-1],))
        encoded = layers.Dense(encoder_neurons[0], activation=encoder_activations[0])(input_shape)
        for i in range(1, len(encoder_neurons)):
            encoded = layers.Dense(encoder_neurons[i], activation=encoder_activations[i])(encoded)

        ##define the decoder
        decoded = layers.Dense(decoder_neurons[0], activation=decoder_activations[0])(encoded)
        for i in range(1, len(decoder_neurons)):
            decoded = layers.Dense(decoder_neurons[i], activation=decoder_activations[i])(decoded)

        super().__init__(input_shape, decoded)


# Initialize an empty list to hold all the profiles
all_profiles = []

for i, dir_name in enumerate(tqdm(dir_names, desc="Importing data from directories")):
    dir_path = os.path.join(cwd, 'StellarTracks', dir_name)

    def extract_number(filename):
        match = re.search(r'\d+', filename)  # find the sequence of digits
        return int(match.group()) if match else float('inf')

    filenames = [filename for filename in os.listdir(dir_path) if re.fullmatch('profile[0-9]+\.data', filename)]
    filenames = sorted(filenames, key=extract_number)  # sort the elements according to the number in the name

    for j, filename in enumerate(tqdm(filenames, desc=f"Importing from {dir_name}", leave=False)):
        filename = os.path.join(dir_path, filename)
        data = mw.read_profile(filename)
        profile_df = pd.DataFrame(data)
        filtered_profile_df = profile_df[column_filter].copy()
        train_filtered_profile_df = profile_df[column_filter_train].copy()

        norm_radius = (filtered_profile_df['radius'] - filtered_profile_df['radius'].min()) / \
                      (filtered_profile_df['radius'].max() - filtered_profile_df['radius'].min())
        norm_profiles = []

        for k, column in enumerate(column_filter_train):
            #norm = (filtered_profile_df[column] - filtered_profile_df[column].min()) / \
            #       (filtered_profile_df[column].max() - filtered_profile_df[column].min())
            norm = filtered_profile_df[column]
            norm = np.asarray(norm.T)
            int_norm = UnivariateSpline(norm_radius, norm, k=2, s=0)(r)
            norm_profiles.append(int_norm)
            #print("Length of this profile: ",len(norm_profiles))

        all_profiles.append(np.array(norm_profiles).T)
        #print("Num profiles: ", len(all_profiles))

# Convert the list of profiles to a numpy array
all_profiles = np.array(all_profiles)
print("Final length of all profiles",len(all_profiles))

# Split the data
x_train, x_test = train_test_split(all_profiles, test_size=0.2)
print(x_train.shape)
print(x_test.shape)

# reshape in the form n_prof*(n_points*n_features)
train_shape=x_train.shape
test_shape=x_test.shape
x_train=np.reshape(x_train,(train_shape[0],train_shape[1]*train_shape[2]),order="F")
x_test=np.reshape(x_test,(test_shape[0],test_shape[1]*test_shape[2]),order="F")
print(x_train.shape)
print(x_test.shape)

for i in range(1, 10):
    ###Autoencoder parameters
    encoder_neurons = [100, i]  # last value should be the latent dimension
    decoder_neurons = encoder_neurons[:len(encoder_neurons) - 1][::-1]  # same inverted values for the hidden layers
    decoder_neurons.append(n_points*len(column_filter_train))  # last value should be original size
    encoder_activations = ['leaky_relu'] * len(encoder_neurons)  # with relu better learning
    decoder_activations = ['leaky_relu'] * (len(encoder_neurons) - 1)  # relu for better learning
    decoder_activations.append('leaky_relu')  # at least the last value should be sigmoid

    autoencoder = Autoencoder(encoder_neurons=encoder_neurons, decoder_neurons=decoder_neurons,
                              encoder_activations=encoder_activations, decoder_activations=decoder_activations)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.summary()

    history = autoencoder.fit(x_train, x_train,
                              epochs=1000,
                              shuffle=True,
                              validation_data=(x_test, x_test),
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

    # Save loss graphs
    file_save_dir = os.path.join(os.getcwd(), "Graphs", f"TrainValLoss_dim_{i}.png")
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f'Training Loss VS Validation Loss - Latent Dim = {i}')
    plt.legend()
    plt.savefig(file_save_dir)
    plt.close()

    x_reconstructed = autoencoder.predict(x_test)
    print(x_reconstructed.shape)

    # Plot and save original vs reconstructed profiles
    n=5
    
    for k,feature in enumerate(column_filter_train):
        plt.figure(figsize=(20, 10))
        plt.axis("off")
        plt.title(f'Examples of fit of feature {feature} with latent dimension = {i}')
        # Display original
        for j in range(n):
            print(x_test)
            print(x_test.shape)
            original_data=x_test[j,k*n_points:(k+1)*n_points]
            print(original_data.shape)
            reconstructed_data=x_reconstructed[j,k*n_points:(k+1)*n_points]
            print(reconstructed_data.shape)
            diff=original_data-reconstructed_data

            ax = plt.subplot(2, n, j + 1)
            plt.scatter(r, original_data,c="blue",label="original")
            plt.scatter(r, reconstructed_data,c="red",label="reconstructed")
            plt.legend()
            plt.gray()

            #Display reconstruction
            ax = plt.subplot(2, n, j + 1 + n)
            plt.scatter(r, diff,label="Original-Reconstructed",c="green")
            plt.legend()
            plt.gray()

        file_save_dir = os.path.join(os.getcwd(), "Graphs", f"OriginalReconstructed_{i}_{feature}.png")
        plt.savefig(file_save_dir)
        plt.close()
