import mesa_web as mw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import os 
import re

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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

### 
# Fare un grid search per le dim degli strati e il padding (error catch per il padding)
# n_points (?)
# trovare la dim_latente migliore
# slope leakyrelu (?fatto)
# cumsum delle distanze tra input e output (come estimator)
# il numero di epoche va bene? 


# define the current working directory
cwd = os.getcwd()  

dir_names=['all'] 
column_filter = ['mass','radius', 'initial_mass', 'initial_z', 'star_age', 'logRho','logT','Teff','energy','photosphere_L', 'photosphere_r', 'star_mass','h1','he3','he4']
column_filter_train = ['mass', 'logRho','logT','energy'] 
n_points=50   # n of points to sample from each profile
r=np.linspace(0, 1, n_points) # n_points equidistant 

if dir_names[0] == 'all':
    dir_names = ['MESA-Web_M07_Z00001', 'MESA-Web_M07_Z002', 'MESA-Web_M10_Z002', 'MESA-Web_M10_Z0001',
                 'MESA-Web_M10_Z00001', 'MESA-Web_M15_Z0001', 'MESA-Web_M15_Z00001', 'MESA-Web_M30_Z00001',
                 'MESA-Web_M30_Z002', 'MESA-Web_M50_Z00001', 'MESA-Web_M50_Z002', 'MESA-Web_M50_Z001',
                 'MESA-Web_M5_Z002', 'MESA-Web_M5_Z0001', 'MESA-Web_M1_Z00001', 'MESA-Web_M1_Z0001']

### Normalization of each column
scalers = {column: MinMaxScaler() for column in column_filter_train}

# Initialize a list to hold all data
all_profiles = []

for i,dir_name in enumerate(tqdm(dir_names, desc="Importing data from directories")):

  dir_name=os.path.join(cwd,'StellarTracks',dir_name)

  def extract_number(filename): # function use to extract the number of the profile
    match = re.search(r'\d+', filename)  #find the sequence of digits
    return int(match.group()) if match else float('inf')

  filenames=[filename for filename in os.listdir(dir_name) if re.fullmatch('profile[0-9]+\.data',filename)]
  filenames=sorted(filenames, key=extract_number) #sort the elements according to the number in the name

  for j,filename in enumerate(tqdm(filenames, desc=f"Importing from {dir_name}", leave=False)):

    filename=os.path.join(dir_name,filename)
    data=mw.read_profile(filename)

    profile_df=pd.DataFrame(data) # DataFrame with all the columns
    filtered_profile_df = profile_df[column_filter].copy()# Create a new DataFrame with only the selected columns
    train_filtered_profile_df = profile_df[column_filter_train].copy() # Create a new DataFrame with only the selected columns for autoencoder training

    ###Normalization process for each feature
    tot_radius=filtered_profile_df['photosphere_r']
    norm_radius=(filtered_profile_df['radius'] - filtered_profile_df['radius'].min())/(tot_radius-filtered_profile_df['radius'].min())
    

    norm_profiles = []

    ### Apply normalization for each training column
    for column in column_filter_train:
            column_values = filtered_profile_df[column].values.reshape(-1, 1)  # Reshape for sklearn
            scalers[column].fit(column_values)  # Fit scaler
            norm = scalers[column].transform(column_values).flatten()  # Normalize data

            # Print the first 50 normalized mass values if the column is 'mass'
            #if column == 'mass':
                #print("Primi 50 valori normalizzati della massa:")
                #print(norm[:50])  # Print the first 50 normalized mass values
        

            norm = np.asarray(norm.T)  # Convert to numpy array
            int_norm = UnivariateSpline(norm_radius, norm, k=2, s=0)(r)  # Interpolate over the normalized radius
            norm_profiles.append(int_norm)

    all_profiles.append(np.array(norm_profiles).T)  # Append normalized profiles
# Convert the list of profiles to a numpy array
all_profiles = np.array(all_profiles)
print("Final length of all profiles",len(all_profiles))
print("Final shape of all profiles:", all_profiles.shape)



x_train, x_test = train_test_split(all_profiles, test_size=0.2, shuffle=False) 
print ('train shape :', x_train.shape) # (train_samples, n_points, num_features)
print ('test shape:', x_test.shape) # (train_samples, n_points, num_features)



# Reshape the data to match the input shape expected by the model
# The input shape is (batch_size, sequence_length, input_dim)
# In our case, sequence_length = 50 and input_dim = n_features

num_features = len(column_filter_train)

x_train_tf = tf.reshape(x_train, ( x_train.shape[0], n_points, num_features))
x_test_tf = tf.reshape(x_test, ( x_test.shape[0], n_points, num_features))

# Print the shapes to verify
print("x_train_tf shape:", x_train_tf.shape) #(1148, 50, 4)
print("x_test_tf shape:", x_test_tf.shape) #(288, 50, 4)

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

# Loop over latent dimensions (2 to 6)
for latent_dim in range(3, 4):################# BEST LATENT DIMENSION = 3
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


    # Plot the original and reconstructed data
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns

    # Plot original vs reconstructed for each feature in the first row
    for i, feature in enumerate(column_filter_train):
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
    for i, feature in enumerate(column_filter_train):
        ax = axes[1, i]
        difference = x_test_tf[0, :, i] - x_reconstructed[0, :, i]
        mean_difference = np.mean(difference)
        ax.scatter(r, difference, label=f'{feature} Difference', color='darkgreen', marker='o')
        ax.axhline(0, color='black', linewidth=0.7)  # Horizontal line at y=0
        ax.axhline(mean_difference, color ='darkred', linestyle='--', linewidth = 1.5, label ='Mean difference')
        ax.set_title(f'{feature} Difference (Latent Dim: {latent_dim})')
        ax.set_xlabel('Normalized Radius')
        ax.set_ylabel(f'{feature} Difference')
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
    plt.grid()
    plt.legend()
    plt.savefig(file_save_dir)
    plt.close()