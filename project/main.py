import mesa_web as mw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 
import re

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from scipy.interpolate import UnivariateSpline

cwd=os.getcwd()
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
    #tot_mass=filtered_profile_df['star_mass']
    #From log to linear scale
    #train_filtered_profile1_df['logRho']=10**(train_filtered_profile1_df['logRho'])
    #train_filtered_profile1_df['logT']=10**(train_filtered_profile1_df['logT'])

    norm_radius=(filtered_profile_df['radius']-filtered_profile_df['radius'].min())/(tot_radius-filtered_profile_df['radius'].min())
    #norm_mass=(filtered_profile_df['mass']-filtered_profile_df['mass'].min())/(tot_mass-filtered_profile_df['mass'].min())
    #norm_rho=(train_filtered_profile1_df['logRho']-train_filtered_profile1_df['logRho'].min())/(train_filtered_profile1_df['logRho'].max()-train_filtered_profile1_df['logRho'].min())
    #norm_t=(train_filtered_profile_df['logT']-train_filtered_profile_df['logT'].min())/(train_filtered_profile_df['logT'].max()-train_filtered_profile_df['logT'].min())
    #norm_energy=(train_filtered_profile_df['energy']-train_filtered_profile_df['energy'].min())/(train_filtered_profile_df['energy'].max()-train_filtered_profile_df['energy'].min())

    norm_radius=np.asarray(norm_radius.T)
    log_rho=np.asarray(train_filtered_profile_df['logRho'].T)
    #norm_mass=np.asarray(norm_mass.T)
    #norm_t=np.asarray(norm_t.T)
    #norm_energy=np.asarray(norm_energy.T)
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
print (x_train.shape)
print (x_test.shape)
print(x_train)
print(x_test)

for i in range(1,10,1):
  latent_dim = i 

  class Autoencoder(Model):
    def __init__(self, latent_dim):
      super(Autoencoder, self).__init__()
      self.latent_dim = latent_dim   
      self.encoder = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(latent_dim, activation='relu'),
      ])
      self.decoder = tf.keras.Sequential([
        layers.Dense(n_points, activation='sigmoid'),
        layers.Reshape((n_points, 1))
      ])

    def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded

  autoencoder = Autoencoder(latent_dim)

  autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

  history = autoencoder.fit(x_train, x_train,
                  epochs=150,
                  shuffle=True,
                  validation_data=(x_test, x_test))

  autoencoder.encoder.summary()

  autoencoder.decoder.summary()

  plt.plot(history.history["loss"], label="Training Loss")
  plt.plot(history.history["val_loss"], label="Validation Loss")
  plt.title(f'Training Loss VS Validation Loss - Latent Dim = %d' % i)
  plt.legend()
  plt.show()
  
  ######################################################################
  ########################## CONVOLUTIONAL #############################
  ######################################################################

import tensorflow as tf
from tensorflow.keras import layers, models

class Network(tf.keras.Model):
    def __init__(self, hyperparameters):
        super(Network, self).__init__()

        # The hyperparameters of the network are saved for reproduction
        self.hyperparameters = hyperparameters
        
        self.input_size = hyperparameters['input_size']
        self.hidden_size = hyperparameters['hidden_size']
        self.n_hidden_layers = hyperparameters['n_hidden_layers']
        self.output_size = hyperparameters['output_size']
        self.activation = hyperparameters['activation']

        if self.activation == 'relu':
            act = layers.ReLU()
        elif self.activation == 'elu':
            act = layers.ELU()
        elif self.activation == 'leakyrelu':
            act = layers.LeakyReLU(0.2)
        elif self.activation == 'sigmoid':
            act = layers.Activation('sigmoid')

        # Define the layers of the network.
        # 1D Convolutional Auto-Encoder

        self.encoder = models.Sequential([
            layers.Conv1D(128, kernel_size=3, strides=2, padding='same', input_shape=(self.input_size, 4)),
            act,
            layers.Conv1D(256, kernel_size=3, strides=2, padding='same'),
            act,
            layers.Conv1D(128, kernel_size=18, strides=1),
            act,
            layers.Conv1D(71, kernel_size=1, strides=1)
        ])

        self.decoder = models.Sequential([
            layers.Conv1DTranspose(128, kernel_size=1, strides=1, input_shape=(None, 71)),
            act,
            layers.Conv1DTranspose(256, kernel_size=18, strides=1),
            act,
            layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', output_padding=1),
            act,
            layers.Conv1DTranspose(4, kernel_size=3, strides=2, padding='same')
        ])

    def forward(self, x):
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

# Example usage
hyperparameters = {
    'input_size': 100,  # Example input size
    'hidden_size': 128,  # Example hidden layer size
    'n_hidden_layers': 3,  # Example number of hidden layers
    'output_size': 10,  # Example output size
    'activation': 'relu'  # Example activation function
}

model = Network(hyperparameters)
model.build(input_shape=(None, hyperparameters['input_size'], 4))  # Example input shape
model.summary()

        