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

### 
# Fare un grid search per le dim degli strati e il padding (error catch per il padding)
# n_points (?)
# trovare la dim_latente migliore
# slope leakyrelu (?fatto)
# cumsum delle distanze tra input e output (come estimator)


# define the current working directory
cwd = os.getcwd()  

dir_names=['MESA-Web_M07_Z00001'] 
column_filter = ['mass','radius', 'initial_mass', 'initial_z', 'star_age', 'logRho','logT','Teff','energy','photosphere_L', 'photosphere_r', 'star_mass','h1','he3','he4']
column_filter_train = ['mass', 'logRho','logT','energy'] 
n_points=50   # n of points to sample from each profile
r=np.linspace(0, 1, n_points) # n_points equidistant 

if dir_names[0] == 'all':
    dir_names = ['MESA-Web_M07_Z00001', 'MESA-Web_M07_Z002', 'MESA-Web_M10_Z002', 'MESA-Web_M10_Z0001',
                 'MESA-Web_M10_Z00001', 'MESA-Web_M15_Z0001', 'MESA-Web_M15_Z00001', 'MESA-Web_M30_Z00001',
                 'MESA-Web_M30_Z002', 'MESA-Web_M50_Z00001', 'MESA-Web_M50_Z002', 'MESA-Web_M50_Z001',
                 'MESA-Web_M5_Z002', 'MESA-Web_M5_Z0001', 'MESA-Web_M1_Z00001', 'MESA-Web_M1_Z0001']

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

    #Normalization process for each feature
    tot_radius=filtered_profile_df['photosphere_r']
    norm_radius=(filtered_profile_df['radius'] - filtered_profile_df['radius'].min())/(tot_radius-filtered_profile_df['radius'].min())
    
    norm_profiles = []
    
    for k, column in enumerate(column_filter_train):
       norm = filtered_profile_df[column]
       norm = np.asarray(norm.T)
       int_norm = UnivariateSpline(norm_radius, norm, k=2, s=0)(r)
       norm_profiles.append(int_norm)

    all_profiles.append(np.array(norm_profiles).T)

# Convert the list of profiles to a numpy array
all_profiles = np.array(all_profiles)
print("Final length of all profiles",len(all_profiles))
print("Final shape of all profiles:", all_profiles.shape)




#print('linear train_df shape:', linear_train_df.shape)
#print(linear_train_df)


x_train, x_test = train_test_split(all_profiles, test_size=0.2)
print ('train shape :', x_train.shape) # (train_samples, n_points, num_features)
print ('test shape:', x_test.shape) # (train_samples, n_points, num_features)
#print(x_train)
#print(x_test)


# Reshape the data to match the input shape expected by the model
# The input shape is (batch_size, sequence_length, input_dim)
# In our case, sequence_length = 50 and input_dim = n_features

num_features = len(column_filter_train)

x_train_tf = tf.reshape(x_train, ( x_train.shape[0], n_points, num_features))
x_test_tf = tf.reshape(x_test, ( x_test.shape[0], n_points, num_features))

# Print the shapes to verify
print("x_train_tf shape:", x_train_tf.shape) #(1148, 50, 4)
print("x_test_tf shape:", x_test_tf.shape) #(288, 50, 4)


#Define all possible padding options
#valid_paddings = ['valid', 'same']
#best_combination = None

#combinations = list(product(valid_paddings, repeat= 7))
#print('combinations:', combinations)

#for combination in combinations:
    #print(f"Trying combination: {combination}")
    #try:

act = tf.keras.layers.LeakyReLU(0.4)

e_layer1 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=2, padding='same', activation=act)(x_train_tf) # same
e_layer2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same', activation=act)(e_layer1) # same
e_layer3 = tf.keras.layers.Conv1D(filters=71, kernel_size=3, strides=2, padding='valid', activation=act)(e_layer2) #valid
e_layer4 = tf.keras.layers.Conv1D(filters=4, kernel_size=3, strides=2, padding='same', activation=act)(e_layer3) # same

print('shape layer1: ', e_layer1.shape, '\n')
print('shape layer2: ', e_layer2.shape, '\n')
print('shape layer3: ', e_layer3.shape, '\n')
print('shape layer4: ', e_layer4.shape, '\n')
print('DECODER:', '\n')

d_layer1 = tf.keras.layers.Conv1DTranspose(filters=71, kernel_size=3, strides=2, padding='same', activation=act)(e_layer4) # same
d_layer2 = tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation=act)(d_layer1) # same 
d_layer3 = tf.keras.layers.Conv1DTranspose(filters=256, kernel_size=3, strides=2, padding='valid', activation=act)(d_layer2) # valid
d_layer4 = tf.keras.layers.Conv1DTranspose(filters=4, kernel_size=3, strides=2, padding='same', activation=act)(d_layer3) # same

print('shape layer1: ', d_layer1.shape, '\n')
print('shape layer2: ', d_layer2.shape, '\n')
print('shape layer3: ', d_layer3.shape, '\n')
print('shape layer4: ', d_layer4.shape, '\n')



    # Make predictions on the test data
    #x_reconstructed = autoencoder.predict(x_test_tf)
    #print('Shape of x_reconstructed:', x_reconstructed.shape)

 