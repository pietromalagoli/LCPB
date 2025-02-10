import mesa_web as mw
import aux
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import os 
import re

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from keras import  losses, layers, activations
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from scipy.interpolate import UnivariateSpline
from keras.datasets import mnist
from keras import backend as K
from tqdm import tqdm 
from itertools import product
from sklearn.preprocessing import MinMaxScaler 
from scikeras.wrappers import KerasClassifier   # Wrapper to use Keras models with scikit-learn (necessary for using GridSearchCV here) 

### 
# Fare un grid search per le dim degli strati e il padding (error catch per il padding)
# n_points (?)
# trovare la dim_latente migliore
# slope leakyrelu (?fatto)
# cumsum delle distanze tra input e output (come estimator)
# il numero di epoche va bene? 


#### PREPROCESSING

x_train_tf, x_test_tf, features, r = aux.preprocess(dir_names=['all'], features=['mass', 'logRho', 'logT', 'energy'], n_points=50)

class Network(tf.keras.Model):
    def __init__(self, hyperparameters):
        super(Network, self).__init__()

        # The hyperparameters of the network are saved for reproduction
        self.hyperparameters = hyperparameters

        self.input_size = hyperparameters['input_size']
        self.output_size = hyperparameters['output_size']
        self.activation = hyperparameters['activation']
        self.latent_dim = hyperparameters['latent_dim']

        if self.activation == 'relu':
            act = tf.keras.layers.ReLU()
        if self.activation == 'elu':
            act = tf.keras.layers.ELU()
        if self.activation == 'leakyrelu':
            act = tf.keras.layers.LeakyReLU(0.4)

        # Define the layers of the network (encoder and decoder)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=2, padding='same', activation=act),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same', activation=act),
            tf.keras.layers.Conv1D(filters=71, kernel_size=3, strides=2, padding='valid', activation=act),
            tf.keras.layers.Conv1D(filters=self.latent_dim, kernel_size=5, strides=2, padding='same', activation=act),
        ]) 

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(filters=71, kernel_size=3, strides=2, padding='same', activation=act),
            tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation=act),
            tf.keras.layers.Conv1DTranspose(filters=256, kernel_size=3, strides=2, padding='valid', activation=act),
            tf.keras.layers.Conv1DTranspose(filters=4, kernel_size=3, strides=2, padding='same', activation=act)
        ])
    
    def call(self, x):
        """
        Evaluate the network

        Parameters
        ----------
        x : tensor
            Input tensor

        Returns
        -------
        tensor
            Output tensor
        """       
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

# Hyperparameters setup
hyperparameters = {
    'input_size': x_train_tf.shape[1:],  # Input shape
    'output_size': x_train_tf.shape[-1],  # Output shape
    'activation': 'leakyrelu',  # Activation function
    'latent_dim': 4  # Latent dimension
}

# Loop over latent dimensions 
for latent_dim in range(3, 5):################# BEST LATENT DIMENSION = 3
    print(f"\nTraining with latent dim = {latent_dim}")
    hyperparameters['latent_dim'] = latent_dim
    autoencoder = Network(hyperparameters)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    # Train the model
    history = autoencoder.fit(x_train_tf, x_train_tf,
                              epochs=100,
                              shuffle=True,
                              validation_data=(x_test_tf, x_test_tf),
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])
    
    #print(f"Latent Dim = {latent_dim}:")
    #print("Loss history:", history.history['loss'])
    #print("Validation Loss history:", history.history['val_loss'])

    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    # Make predictions on the test data
    x_reconstructed = autoencoder.predict(x_test_tf)
    print('Shape of x_reconstructed:', x_reconstructed.shape)
    print("Test shpe:", x_test_tf.shape)

    print("#################################################")
    print("Total loss:",autoencoder.compute_loss(x_test_tf, x_test_tf, x_reconstructed).numpy())

    ##########################################################################################
    ################################## GRID SEARCH ###########################################
    ##########################################################################################
    '''
    # Define the parameter sets
    kernel_size = [2,3,4,5]
    stride = [1,2,3]
    padding = 'causal'
    param_grid={'kernel_size': kernel_size, \
                'stride': stride,
                'padding': padding}
    
    # Use a wrapper to use the Keras model with scikit-learn
    keras_clf = KerasClassifier(build_fn=autoencoder, epochs=100, batch_size=32, metrics= 'Accuracy', \
                                loss=tf.keras.losses.MeanSquaredError(), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)], \
                                verbose=1)

    # Implement the grid search
    gridsearch = GridSearchCV(estimator=keras_clf, param_grid=param_grid ,cv=5)
    gridsearch.fit(x_train_tf, x_train_tf)
    print('############### GRID SEARCH RESULTS ################')
    gridsearch.cv_results_
    print('Best parameters:', gridsearch.best_params_)
    '''
    # For the padding we retrieve its value from the formula: O=[(I−K+2P)/S]+1 (O: output size, I: input size, K: kernel size, P: padding, S: stride) 
    # (maybe you have to apply the floor function on it) (for the padding P I think is P=0 for padding 'valid' and P=1 for padding 'same')
    # So it's P = [S * (O + 1) - I + K] / 2
    
    def evaluate_padding(I,K,P,S):
        return ((I-K+2*P)/S)+1
    
    # COME GLI FACCIO A DARE IL VALORE DI PADDING NEL GRID SEARCH SE DALLA FORMULA ESCE 0 O 1 E IO GLI DEVO DARE VALID O SAME??

    # Plot the original and reconstructed data
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns

    # Plot original vs reconstructed for each feature in the first row
    for i, feature in enumerate(features):
        ax = axes[0, i]
        ax.scatter(r, x_test_tf[0, :, i], label='Original', color='blue', marker='o')
        ax.scatter(r, x_reconstructed[0, :, i], label='Reconstructed', color='red', marker='x')
        ax.set_title(f'{feature} (Latent Dim: {latent_dim})')
        ax.set_xlabel('Normalized Radius')
        ax.set_ylabel(f'{feature}')
        ax.grid(True)
        if i == 0:
            ax.legend(loc='best')

    # Plot the difference between original and reconstructed for each feature in the second row
    for i, feature in enumerate(features):
        ax = axes[1, i]
        difference = np.abs(x_test_tf[0, :, i] - x_reconstructed[0, :, i])
        mean_difference = np.mean(difference)
        ax.scatter(r, difference, label=f'{feature} absolute difference', color='darkgreen', marker='o')
        ax.axhline(0, color='black', linewidth=0.7)  # Horizontal line at y=0
        ax.axhline(mean_difference, color ='darkred', linestyle='--', linewidth = 1.5, label ='Mean absolute difference')
        ax.set_title(f'{feature} absolute difference (Latent Dim: {latent_dim})')
        ax.set_xlabel('Normalized Radius')
        ax.set_ylabel(f'{feature} absolute difference')
        ax.grid(True) 
    
    # Adjust layout
    plt.tight_layout()

    file_save_dir = os.path.join(os.getcwd(), "Graphs", f"LatentDim_{latent_dim}_Comparison.png")
    plt.savefig(file_save_dir)
    plt.close()

    # Save loss graphs
    file_save_dir = os.path.join(os.getcwd(), "Graphs", f"TrainValLoss_dim_{latent_dim}.png")
    plt.plot(history.history["loss"], label="Training Loss", color='orange')
    plt.plot(history.history["val_loss"], label="Validation Loss", color='blue')
    plt.title(f'Training Loss VS Validation Loss - Latent Dim = {latent_dim}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig(file_save_dir)
    plt.close()