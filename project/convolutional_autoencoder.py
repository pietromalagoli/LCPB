import mesa_web as mw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 
import re

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import  losses, layers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from scipy.interpolate import UnivariateSpline
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K



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


for i,dir_name in enumerate(dir_names):

  print(f"####\t\tIMPORTING DATA FROM FOLDER {dir_name}\t\t####")
  dir_name=os.path.join(cwd,'StellarTracks',dir_name)

  def extract_number(filename):
    match = re.search(r'\d+', filename)  #find the sequence of digits
    return int(match.group()) if match else float('inf')

  filenames=[filename for filename in os.listdir(dir_name) if re.fullmatch('profile[0-9]+\.data',filename)]
  filenames=sorted(filenames, key=extract_number) #sort the elements according to the number in the name

  for j,filename in enumerate(filenames):

    print(f"####\t\t\tIMPORTING FILE {filename}\t\t\t####")  
    filename=os.path.join(dir_name,filename)

    data=mw.read_profile(filename)

    profile_df=pd.DataFrame(data)

    # Create a new DataFrame with only the selected columns
    filtered_profile_df = profile_df[column_filter].copy()
    # Create a new DataFrame with only the selected columns for autoencoder training
    train_filtered_profile_df = profile_df[column_filter_train].copy()

    #Normalization process
    tot_radius=filtered_profile_df['photosphere_r']

    norm_radius=(filtered_profile_df['radius']-filtered_profile_df['radius'].min())/(tot_radius-filtered_profile_df['radius'].min())

    norm_radius=np.asarray(norm_radius.T)
    log_rho=np.asarray(train_filtered_profile_df['logRho'].T)
    
    int_log_rho=UnivariateSpline(norm_radius,log_rho,k=2,s=0)(r)

    train_df=pd.DataFrame(data=int_log_rho.T,columns=[f"log_rho_{i}_{j}"])  #in the format _indexOfFolder_IndexOfProfile
  
    if (i==0 and j==0):
      linear_train_df=train_df
    else:
      linear_train_df=pd.concat([linear_train_df,train_df],axis=1)

print(linear_train_df.shape)
print(linear_train_df)

#for i in range(linear_train_df.shape[1]):
#  plt.plot(r,linear_train_df[f"log_rho_{i}"])
#  plt.show()

x_train, x_test = train_test_split(linear_train_df.T, test_size=0.2)
print ('train shape :', x_train.shape) # (69, 50)
print ('test shape:', x_test.shape) #(18, 50)
print(x_train)
print(x_test)


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
        
        #self.encoder = tf.keras.Sequential([
            #tf.keras.layers.Conv1D(filters=500, kernel_size=3, strides=2, padding='same', activation=act), # 1D 128 kernels of length=36  ##filters -> dimesion of output space
            #tf.keras.layers.Conv1D(filters=250, kernel_size=3, strides=2, padding='same', activation=act), # 1D 4*64 kernels of length=18
            #tf.keras.layers.Conv1D(filters=75, kernel_size=3, strides=1, activation=act), # 1D 128 kernels of length=1
            #tf.keras.layers.Conv1D(filters=self.latent_dim, kernel_size=1, strides=1) # NiN 
            
        #])

        #self.decoder = tf.keras.Sequential([
            #tf.keras.layers.Conv1DTranspose(filters=self.latent_dim, kernel_size=1, strides=1, activation=act), # Reverse of NiN
            #tf.keras.layers.Conv1DTranspose(filters=75, kernel_size=3, strides=1, activation=act), # Reverse of kernel 3
            #tf.keras.layers.Conv1DTranspose(filters=250, kernel_size=3, strides=2, padding='same', activation=act), # Reverse of kernel 3 and stride 2
            #tf.keras.layers.Conv1DTranspose(filters=500, kernel_size=3, strides=2, padding='same') # Reverse of kernel 3 and stride 2
        #])

        self.encoder_conv1 = tf.keras.layers.Conv1D(filters= 32, kernel_size = 3, strides = 2, padding= 'valid', activation=self.activation)
        self.encoder_conv2 = tf.keras.layers.Conv1D(filters=self.latent_dim, kernel_size=3, strides=2, padding='same', activation=self.activation)
        
        self.decoder_conv1 = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding='valid', activation=self.activation)
        self.decoder_conv2 = tf.keras.layers.Conv1DTranspose(filters=self.output_size, kernel_size=3, strides=2, padding='same', activation=None)

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
        encoded = self.encoder_conv1(x)
        encoded = self.encoder_conv2(encoded)
        
        decoded= self.decoder_conv1(encoded)
        decoded = self.decoder_conv2(decoded)
        return decoded
    
    


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
                  epochs=50,
                  shuffle=True,
                  validation_data=(x_test_tf, x_test_tf))

autoencoder.encoder.summary()

autoencoder.decoder.summary()

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title(f'Training Loss VS Validation Loss - Latent Dim = %d' % i)
plt.legend()
plt.show()