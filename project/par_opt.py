# Libraries
import mesa_web as mw
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm 
import ast

import warnings
# Suppress warnings
warnings.filterwarnings('ignore')

# Define the current working directory
cwd = os.getcwd()  

# Initialize the list of directory names to 'all'
dir_names=['all']   
# Define the columns to be filtered from the data and the columns to be used for training
column_filter = ['mass','radius', 'initial_mass', 'initial_z', 'star_age', 'logRho','logT','Teff','energy','photosphere_L', 'photosphere_r', 'star_mass','h1','he3','he4']
column_filter_train = ['mass', 'logRho','logT','energy'] 
# Number of points to sample from each profile
n_points=50   
# Generate n_points equidistant points between 0 and 1
r=np.linspace(0, 1, n_points)  

if dir_names[0] == 'all':
    dir_names = ['MESA-Web_M07_Z00001', 'MESA-Web_M07_Z002', 'MESA-Web_M10_Z002', 'MESA-Web_M10_Z0001',
                 'MESA-Web_M10_Z00001', 'MESA-Web_M15_Z0001', 'MESA-Web_M15_Z00001', 'MESA-Web_M30_Z00001',
                 'MESA-Web_M30_Z002', 'MESA-Web_M50_Z00001', 'MESA-Web_M50_Z002', 'MESA-Web_M50_Z001',
                 'MESA-Web_M5_Z002', 'MESA-Web_M5_Z0001', 'MESA-Web_M1_Z00001', 'MESA-Web_M1_Z0001']
    

# Initialize a list to hold all data
all_profiles = []

# Loop through each directory in dir_names
for i,dir_name in enumerate(tqdm(dir_names, desc="Importing data from directories")):

  # Construct the full path to the directory
  dir_name=os.path.join(cwd,'StellarTracks',dir_name)
  
  # Function to extract the number from the filename
  def extract_number(filename):
    match = re.search(r'\d+', filename)  # Find the sequence of digits
    return int(match.group()) if match else float('inf')

  # List all files in the directory that match the pattern 'profile[0-9]+.data' and sort them by the number in the filename
  filenames=[filename for filename in os.listdir(dir_name) if re.fullmatch('profile[0-9]+\.data',filename)]
  filenames=sorted(filenames, key=extract_number) 

  # Loop through each file in the sorted list
  for j,filename in enumerate(tqdm(filenames, desc=f"Importing from {dir_name}", leave=False)):

    filename=os.path.join(dir_name,filename)
    data=mw.read_profile(filename)

    # Create the dataframe for training
    profile_df=pd.DataFrame(data) # DataFrame with all the columns
    filtered_profile_df = profile_df[column_filter].copy()# Create a new DataFrame with only the selected columns
    train_filtered_profile_df = profile_df[column_filter_train].copy() # Create a new DataFrame with only the selected columns for autoencoder training

    # Normalization of the radius
    norm_radius=(filtered_profile_df['radius'] - filtered_profile_df['radius'].min())/(filtered_profile_df['radius'].max()-filtered_profile_df['radius'].min())
    
    # Initialize a list to hold the normalized profiles
    norm_profiles = []

    # Loop through each column in the filtered profile
    for column in column_filter_train:
            
            norm = filtered_profile_df[column]
            
            # Take the log of the energy column
            if column == 'energy':
                norm = pd.Series([np.log(x) for x in filtered_profile_df[column]])
                column_name= 'logEnergy'
            else: 
                column_name = column
            
            # Interpolate the normalized radius
            norm = np.asarray(norm.T)
            int_norm = UnivariateSpline(norm_radius, norm, k=2, s=0)(r)  
            norm_profiles.append((column_name, int_norm)) 

    final_profile= {name: values for name, values in norm_profiles}
    all_profiles.append(np.array(list(final_profile.values())).T)  # Append normalized profiles


# Convert the list of profiles to a numpy array
all_profiles = np.array(all_profiles)

# Split the data into training and testing sets
x_train, x_test = train_test_split(all_profiles, test_size=0.2, shuffle=False)

# Reshape the data to match the input shape expected by the model
num_features = len(column_filter_train)

x_train_tf = tf.reshape(x_train, ( x_train.shape[0], n_points, num_features))
x_test_tf = tf.reshape(x_test, ( x_test.shape[0], n_points, num_features))

# Define the network class
class Network(tf.keras.Model):
    # Define the layers of the network
    def __init__(self, hyperparameters):
        # Initialize the parent class
        super(Network, self).__init__()

        # The hyperparameters of the network are saved for reproduction
        self.hyperparameters = hyperparameters

        # Extract the hyperparameters
        self.input_size = hyperparameters['input_size']
        self.output_size = hyperparameters['output_size']
        self.activation = hyperparameters['activation']
        self.latent_dim = hyperparameters['latent_dim']

        if self.activation == 'leakyrelu': 
            act = tf.keras.layers.LeakyReLU(alpha= 0.35)
        else:
            act = self.activation

        # Define the layers of the network (encoder and decoder)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=2, padding='same', activation=act),  # same
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same', activation=act),   # same
            tf.keras.layers.Conv1D(filters=71, kernel_size=3, strides=2, padding='valid', activation=act),   # valid 
            tf.keras.layers.Conv1D(filters=self.latent_dim, kernel_size=3, strides=2, padding='same', activation=act),   # same
        ]) 

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(filters=71, kernel_size=3, strides=2, padding='same', activation=act),  # same
            tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation=act),  # same
            tf.keras.layers.Conv1DTranspose(filters=256, kernel_size=3, strides=2, padding='valid', activation=act),   # valid
            tf.keras.layers.Conv1DTranspose(filters=4, kernel_size=3, strides=2, padding='same', activation=act)  # same
        ])
    
    # Define the call method for the Network class
    def call(self, x):
        # Pass the input through the encoder to get the encoded representation  
        encoded = self.encoder(x)
        # Pass the encoded representation through the decoder to reconstruct the input
        decoded = self.decoder(encoded)

        # Return the reconstructed output
        return decoded

##########
# PARAMETERS OPTIMIZATION

save_dir = os.path.join(os.getcwd(), "Graphs")  

optimizers=['adam','nadam','rmsprop','adagrad','adadelta','ftrl','adamax','adamw','lion']
activations=['leaky_relu','relu','gelu','softplus','elu','selu','silu']

# activations (fixed adam)
performance_data=pd.DataFrame()

optimizer = optimizers[0]

for activation in activations:

    # Hyperparameters setup
    hyperparameters = {
        'input_size': x_train_tf.shape[1:],  # Input shape
        'output_size': x_train_tf.shape[-1],  # Output shape
        'activation': activation,  # Fixed activation function
        'latent_dim': 4  # Fixed latent dimension
    }

    autoencoder = Network(hyperparameters)

    # Compile the model
    autoencoder.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    # Train the model
    history = autoencoder.fit(x_train_tf, x_train_tf,
                                epochs=150,
                                shuffle=True,
                                validation_data=(x_test_tf, x_test_tf),
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])
    
    new_row=pd.DataFrame([{'activation':activation,
                           'optimizer':optimizer,
                            'avg_final_val_loss': np.mean(history.history['loss'][-15:])}])
    performance_data=pd.concat([performance_data,new_row])

performance_data.to_csv(f'results-{optimizer}-{hyperparameters['latent_dim']}.csv')

plot_type='activation'  #activation or optimizer
title=f'Activations performances with fixed optimizer as {optimizer}'

file=f'results-{optimizer}-{hyperparameters['latent_dim']}.csv'

fig = plt.figure(figsize=(10,5))

data=pd.read_csv(file)
'''
data['avg_final_val_loss'] = data['avg_final_val_loss'].apply(lambda x: ast.literal_eval(x))
data['avg_final_val_loss'] = data['avg_final_val_loss'].apply(lambda x: pd.Series(x).mean())
'''
colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))
for i, (x, y) in enumerate(zip(data[plot_type], data['avg_final_val_loss'])):
    plt.scatter(x, y, color=colors[i])
    
plt.title(title)
plt.xlabel(plot_type)
plt.ylabel('Final Loss (mean over last 15 steps)')
file_path = os.path.join(save_dir,f'{plot_type}_opt{hyperparameters['latent_dim']}.png')
plt.savefig(file_path)


# optimizers computation (fixed ReLU)
performance_data=pd.DataFrame()

activation = activations[0]
   
for optimizer in optimizers:
 
    # Hyperparameters setup
    hyperparameters = {
        'input_size': x_train_tf.shape[1:],  # Input shape
        'output_size': x_train_tf.shape[-1],  # Output shape
        'activation': activation,  # Fixed activation function
        'latent_dim': 4  # Fixed latent dimension
    }

    autoencoder = Network(hyperparameters)

    # Compile the model
    autoencoder.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    # Train the model
    history = autoencoder.fit(x_train_tf, x_train_tf,
                                epochs=150,
                                shuffle=True,
                                validation_data=(x_test_tf, x_test_tf),
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])
    
    new_row=pd.DataFrame([{'activation':activation,
                           'optimizer':optimizer,
                            'avg_final_val_loss': np.mean(history.history['loss'][-15:])}])
    performance_data=pd.concat([performance_data,new_row])

performance_data.to_csv(f'results-{activation}-{hyperparameters['latent_dim']}.csv')


plot_type='optimizer'  #activation or optimizer
title=f'Optimizers performances with fixed optimizer as {activation}'

file=f'results-{activation}-{hyperparameters['latent_dim']}.csv'

fig = plt.figure(figsize=(10,5))

data=pd.read_csv(file)
'''
data['avg_final_val_loss'] = data['avg_final_val_loss'].apply(lambda x: ast.literal_eval(x))
data['avg_final_val_loss'] = data['avg_final_val_loss'].apply(lambda x: pd.Series(x).mean())
'''
colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))
for i, (x, y) in enumerate(zip(data[plot_type], data['avg_final_val_loss'])):
    plt.scatter(x, y, color=colors[i])
    
plt.title(title)
plt.xlabel(plot_type)
plt.ylabel('Final Loss (mean over last 15 steps)')
file_path = os.path.join(save_dir,f'{plot_type}_opt{hyperparameters['latent_dim']}.png')
plt.savefig(file_path)