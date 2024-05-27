import mesa_web as mw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 
import re

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from keras import  losses, layers, activations
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from scipy.interpolate import UnivariateSpline
from keras.datasets import mnist
from keras import backend as K
from tqdm import tqdm



# define the current working directory
cwd = os.getcwd()  

# Define the Stellar Tracks directory
#stellar_tracks_dir = os.path.join(cwd, 'StellarTracks')

#all_entries = os.listdir(stellar_tracks_dir)
#pattern = re.compile(r'MESA-Web_M\d+_Z\d+')

# List all entries that match the pattern
#dir_names = [d for d in all_entries if pattern.match(d)]
#print("Filtered directory names:")
#print(dir_names)

dir_names=['MESA-Web_M07_Z00001','MESA-Web_M1_Z0001']
column_filter = ['mass','radius', 'initial_mass', 'initial_z', 'star_age', 'logRho','logT','Teff','energy','photosphere_L', 'photosphere_r', 'star_mass','h1','he3','he4']
column_filter_train = ['mass','radius', 'logRho','logT','energy','h1','he3','he4']
n_points=50
r=np.linspace(0,1,n_points)


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
            norm = (filtered_profile_df[column] - filtered_profile_df[column].min()) / \
                   (filtered_profile_df[column].max() - filtered_profile_df[column].min())
            norm = np.asarray(norm.T)
            int_norm = UnivariateSpline(norm_radius, norm, k=2, s=0)(r)
            norm_profiles.append(int_norm)
            print("Length of this profile: ",len(norm_profiles))

        all_profiles.append(np.array(norm_profiles).T)
        print("Num profiles: ", len(all_profiles))

# Convert the list of profiles to a numpy array
all_profiles = np.array(all_profiles)
print("Final length of all profiles",len(all_profiles))

# Split the data
x_train, x_test = train_test_split(all_profiles, test_size=0.2)
print(x_train.shape)
print(x_test.shape)


# Reshape the data to match the input shape expected by the model
# The input shape is (batch_size, sequence_length, input_dim)
# In uor case, sequence_length = 50 and input_dim = n_features

x_train_tf = tf.reshape(x_train, ( x_train.shape[0], 50, 1))
x_test_tf = tf.reshape(x_test, ( x_test.shape[0], 50,1))

# Print the shapes to verify
print("x_train_tf shape:", x_train_tf.shape)
print("x_test_tf shape:", x_test_tf.shape)





class Network(tf.keras.Model):
    def __init__(self, hyperparameters):
        super(Network, self).__init__()

        # The hyperparameters of the network are saved for reproduction
        self.hyperparameters = hyperparameters
        
        self.input_size = hyperparameters['input_size']
        self.hidden_size = hyperparameters['hidden_size']
        self.output_size = hyperparameters['output_size']
        self.activation = hyperparameters['activation']
        self.latent_dim = hyperparameters['latent_dim']

        if (self.activation == 'relu'):
            act = tf.keras.layers.ReLU()
        if (self.activation == 'elu'):
            act = tf.keras.layers.ELU()
        if (self.activation == 'leakyrelu'):
            act = tf.keras.layers.LeakyReLU(0.2)

        # Define the layers of the network ( encoder and decoder)
        
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=500, kernel_size=3, strides=2, padding='valid', activation=act), # 1D 128 kernels of length=36  ##filters -> dimesion of output space
            tf.keras.layers.Conv1D(filters=250, kernel_size=3, strides=2, padding='same', activation=act),
            tf.keras.layers.Conv1D(filters=70, kernel_size=3, strides=2, padding='same', activation=act),
            tf.keras.layers.Conv1D(filters=self.latent_dim, kernel_size=3, strides=2, padding='same', activation=act), # 1D 4*64 kernels of length=18
            #tf.keras.layers.Conv1D(filters=75, kernel_size=3, strides=1, activation=act), # 1D 128 kernels of length=1
            #tf.keras.layers.Conv1D(filters=self.latent_dim, kernel_size=1, strides=1) # NiN 
            
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(filters=70, kernel_size=3, strides=2,padding= 'same', activation=act), # Reverse of NiN
            tf.keras.layers.Conv1DTranspose(filters=250, kernel_size=3, strides=2,padding= 'same', activation=act),
            tf.keras.layers.Conv1DTranspose(filters=500, kernel_size=3, strides=2,padding= 'valid', activation=act),
            tf.keras.layers.Conv1DTranspose(filters=self.output_size, kernel_size=3, strides=2, padding= 'same', activation=activations.sigmoid), # Reverse of kernel 3
            #tf.keras.layers.Conv1DTranspose(filters=250, kernel_size=3, strides=2, padding='same', activation=act), # Reverse of kernel 3 and stride 2
            #tf.keras.layers.Conv1DTranspose(filters=500, kernel_size=3, strides=2, padding='same') # Reverse of kernel 3 and stride 2
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
        decoded= self.decoder(encoded)
        
        return decoded
    
    

#for lat_dim in range(1,10,1):
hyperparameters = {
'input_size' : x_train_tf.shape[1:],
'hidden_size' : 256,  # dimension of hidden layers
'n_hidden_layers': 3,  # number of hidden layers
'output_size': x_train_tf.shape[-1],
'activation': 'relu',
'latent_dim': 4
}
autoencoder = Network(hyperparameters)

# compile
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

#training 
history = autoencoder.fit(x_train_tf, x_train_tf,
                epochs=100,
                shuffle=True,
                validation_data=(x_test_tf, x_test_tf))

autoencoder.encoder.summary()

autoencoder.decoder.summary()

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title(f'Training Loss VS Validation Loss - Latent Dim = %d' % i)
plt.legend()
plt.show()

x_reconstructed = autoencoder.predict(x_test_tf)
print(x_reconstructed.shape)

x_test = x_test_tf.numpy()

# Plot and save original vs reconstructed profiles
n = 10  # How many profiles we will display
plt.figure(figsize=(20, 4))
plt.axis("off")
#plt.title(f'Examples of fit with latent dimension = {i}')
for j in range(n):
    # Display original
    ax = plt.subplot(2, n, j + 1)
    plt.scatter(r, x_test[j].reshape(-1, n_points)[0])
    plt.gray()

    # Display reconstruction
    ax = plt.subplot(2, n, j + 1 + n)
    plt.scatter(r, x_reconstructed[j].reshape(-1, n_points)[0])
    plt.gray()

plt.show()

