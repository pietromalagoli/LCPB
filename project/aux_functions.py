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

def get_data(dir_names,column_filter,column_filter_train,r):
    cwd = os.getcwd()
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

    return all_profiles


def train_autoencoder(all_profiles,encoder_neurons_in,activation,optimizer,loss,plot_loss,plot_reconstructed,save_model,folder,column_filter_train):

    # Split the data
    x_train, x_test = train_test_split(all_profiles, test_size=0.2)
    print(x_train.shape)
    print(x_test.shape)

    # reshape in the form n_prof*(n_points*n_features)
    train_shape=x_train.shape
    test_shape=x_test.shape

    # get info before reshaping

    num_features=train_shape[2]
    n_points=train_shape[1]

    # reshape
    x_train=np.reshape(x_train,(train_shape[0],train_shape[1]*train_shape[2]),order="F")
    x_test=np.reshape(x_test,(test_shape[0],test_shape[1]*test_shape[2]),order="F")
    print(x_train.shape)
    print(x_test.shape)

    for i in range(1, 6):
        ###Autoencoder parameters
        encoder_neurons = encoder_neurons_in.copy()  # last value should be the latent dimension
        encoder_neurons.append(i)
        decoder_neurons = encoder_neurons[:len(encoder_neurons)-1][::-1]  # same inverted values for the hidden layers
        decoder_neurons.append(n_points*num_features)  # last value should be original size
        encoder_activations = [activation] * (len(encoder_neurons))  # with relu better learning
        decoder_activations = [activation] * (len(encoder_neurons))  # relu for better learning
        #decoder_activations.append(activation)  # at least the last value should be sigmoid

        autoencoder = Autoencoder(encoder_neurons=encoder_neurons, decoder_neurons=decoder_neurons,
                                encoder_activations=encoder_activations, decoder_activations=decoder_activations)

        autoencoder.compile(optimizer=optimizer, loss=loss)
        autoencoder.summary()

        history = autoencoder.fit(x_train, x_train,
                                epochs=1000,
                                shuffle=True,
                                validation_data=(x_test, x_test),
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
        
        if plot_loss:
            aux.plot_loss(history,i,folder)

        if plot_reconstructed:
            aux.plot_reconstructed(autoencoder,x_test,column_filter_train,n_points,i,folder)

        if save_model:
            aux.save_model(autoencoder,i,folder)
        
    return autoencoder

def plot_loss(history,i,folder):
    # Save loss graphs
    if not os.path.exists(folder):
        os.mkdir(folder)
    file_save_dir = os.path.join(os.getcwd(), folder, f"TrainValLoss_dim_{i}.png")
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f'Training Loss VS Validation Loss - Latent Dim = {i}')
    plt.legend()
    plt.savefig(file_save_dir)
    plt.close()

def plot_reconstructed(autoencoder,x_test,column_filter_train,n_points,i,folder):
    x_reconstructed = autoencoder.predict(x_test)
    r = np.linspace(0, 1, n_points + 1)[1:]
    # Plot and save original vs reconstructed profiles
    n=5
    
    for k,feature in enumerate(column_filter_train):
        plt.figure(figsize=(20, 10))
        plt.axis("off")
        plt.title(f'Examples of fit of feature {feature} with latent dimension = {i}')
        # Display original
        for j in range(n):
            original_data=x_test[j,k*n_points:(k+1)*n_points]
            reconstructed_data=x_reconstructed[j,k*n_points:(k+1)*n_points]
            rel_diff=(reconstructed_data-original_data)/original_data

            ax = plt.subplot(2, n, j + 1)
            plt.scatter(r, original_data,c="blue",label="original")
            plt.scatter(r, reconstructed_data,c="red",label="reconstructed")
            plt.legend()
            plt.gray()

            #Display reconstruction
            ax = plt.subplot(2, n, j + 1 + n)
            plt.scatter(r, rel_diff,label="Original-Reconstructed",c="green")
            plt.legend()
            plt.gray()

        file_save_dir = os.path.join(os.getcwd(), folder, f"OriginalReconstructed_{i}_{feature}.png")
        plt.savefig(file_save_dir)
        plt.close()

def save_model(model,i,folder):
    # Save trained model
    file_save_dir = os.path.join(os.getcwd(), folder, f"Trained_model_{i}.keras")
    model.save(file_save_dir)