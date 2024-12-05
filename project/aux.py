# Libraries
import mesa_web as mw   # here the pre-preprocessing is carried out
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import re

from sklearn.model_selection import train_test_split
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm 


############################################################################################################
######################################### PREPROCESSING ####################################################
############################################################################################################

def extract_number(filename):
    """Function to extract the number of the file from the filename

    Args:
        filename (str): name of the file to extract the number from

    Returns:
        int: number associated to the file
    """
    
    match = re.search(r'\d+', filename)  # Find the sequence of digits
    return int(match.group()) if match else float('inf')


def preprocess(dir_names: list=['all'], features: list=['mass', 'logRho','logT','energy'], n_points: int=50):
    """This function executes all the preprocessing, returning the train and test datasets

    Args:
        dir_names (list, optional): directory where to take the data from. Defaults to 'all'.
        features (list, optional): features to analyse. Defaults to ['mass', 'logRho','logT','energy'].
        n_points (int, optional): number of points to take for each feature from each file (interpolated). Defaults to 50.

    Returns:
        x_train_tf (tf.Tensor): train set (converted to tensorflow.Tensor).
        x_test_tf (tf.Tensor): test set (converted to tensorflow.Tensor).
        features (list): features list AFTER the preprocessing.
        r_points (np.array): array of equidistant n_points points.
        
    Note: the supported values for dir_names (directories) are (and their combinations): 
                    ['all', 'MESA-Web_M07_Z00001', 'MESA-Web_M07_Z002', 'MESA-Web_M10_Z002', 'MESA-Web_M10_Z0001',
                    'MESA-Web_M10_Z00001', 'MESA-Web_M15_Z0001', 'MESA-Web_M15_Z00001', 'MESA-Web_M30_Z00001',
                    'MESA-Web_M30_Z002', 'MESA-Web_M50_Z00001', 'MESA-Web_M50_Z002', 'MESA-Web_M50_Z001',
                    'MESA-Web_M5_Z002', 'MESA-Web_M5_Z0001', 'MESA-Web_M1_Z00001', 'MESA-Web_M1_Z0001']
                    
          the supported values for features (in input) are (and their combinations):
                    ['mass', 'logRho','logT','energy']
    """
    
    # Define the current working directory
    cwd = os.getcwd()  

    # Define the columns to be filtered from the data and the columns to be used for training
    column_filter = ['mass','radius', 'initial_mass', 'initial_z', 'star_age', 'logRho','logT','Teff','energy','photosphere_L', 'photosphere_r', 'star_mass','h1','he3','he4'] 
    # Generate n_points equidistant points between 0 and 1
    r_points=np.linspace(0, 1, n_points)  

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
        train_filtered_profile_df = profile_df[features].copy() # Create a new DataFrame with only the selected columns for autoencoder training

        # Normalization of the radius
        norm_radius=(filtered_profile_df['radius'] - filtered_profile_df['radius'].min())/(filtered_profile_df['radius'].max()-filtered_profile_df['radius'].min())
        
        # Initialize a list to hold the normalized profiles
        norm_profiles = []

        # Loop through each column in the filtered profile
        for column in features:

            # Compute the norm of the column
            norm = filtered_profile_df[column]
            
            # Take the log of the energy column
            if column == 'energy':
                norm = pd.Series([np.log(x) for x in filtered_profile_df[column]])
                column = 'logEnergy'
                
            # Interpolate the normalized radius
            norm = np.asarray(norm.T)
            int_norm = UnivariateSpline(norm_radius, norm, k=2, s=0)(r_points)  
            norm_profiles.append(int_norm)

        all_profiles.append(np.array(norm_profiles).T)  # Append normalized profiles
        
    # Rename the features accordinlgy to the changes
    features = ['mass', 'logRho','logT','logEnergy']

    # Convert the list of profiles to a numpy array
    all_profiles = np.array(all_profiles)
    print("Final length of all profiles",len(all_profiles))
    print("Final shape of all profiles:", all_profiles.shape)

    # Split the data into training and testing sets
    x_train, x_test = train_test_split(all_profiles, test_size=0.2, shuffle=False)
    print ('train shape :', x_train.shape)   # (train_samples, n_points, num_features)
    print ('test shape:', x_test.shape)      # (train_samples, n_points, num_features)

    # Reshape the data to match the input shape expected by the model
    num_features = len(features)

    x_train_tf = tf.reshape(x_train, ( x_train.shape[0], n_points, num_features))
    x_test_tf = tf.reshape(x_test, ( x_test.shape[0], n_points, num_features))

    # Print the shapes to verify
    print("x_train_tf shape:", x_train_tf.shape) #(1148, 50, 4)
    print("x_test_tf shape:", x_test_tf.shape)   #(288, 50, 4)
    
    return x_train_tf, x_test_tf, features, r_points

############################################################################################################
########################################### MODEL ##########################################################
############################################################################################################


class Network(tf.keras.Model):

    # Define the layers of the network
    def __init__(self, hyperparameters):
        """Initializes the network.

        Args:
            hyperparameters (dict): dictionary of hyperparameters used in the network.
        """
        # Initialize the parent class
        super(Network, self).__init__()

        # The hyperparameters of the network are saved for reproduction
        self.hyperparameters = hyperparameters

        # Extract the hyperparameters
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
            tf.keras.layers.Conv1D(filters=self.latent_dim, kernel_size=3, strides=2, padding='same', activation=act),
        ]) 

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(filters=71, kernel_size=3, strides=2, padding='same', activation=act),
            tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation=act),
            tf.keras.layers.Conv1DTranspose(filters=256, kernel_size=3, strides=2, padding='valid', activation=act),
            tf.keras.layers.Conv1DTranspose(filters=4, kernel_size=3, strides=2, padding='same', activation=act)
        ])
    
    def call(self, x):
        """Call method for the network class.

        Args:
            x (tf.Tensor): input data.

        Returns:
            tf.Tensor: decoded output.
        """
        # Pass the input through the encoder to get the encoded representation  
        encoded = self.encoder(x)
        # Pass the encoded representation through the decoder to reconstruct the input
        decoded = self.decoder(encoded)

        # Return the reconstructed output
        return decoded
