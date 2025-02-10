import mesa_web as mw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import os 
import re

from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from keras import  losses, layers, activations
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from scipy.interpolate import UnivariateSpline
from keras.datasets import mnist
from keras import backend as K




###
# salvare i grafici nella catella in modo da averli tutti insieme
# aumentare le epoche  (early stopping dopo 10 volte che la loss è ugaule si ferma)( nel fit della rete, callback)
# provare con tutte le cartelle 

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

dir_names=['MESA-Web_M07_Z00001', 'MESA-Web_M07_Z002', 'MESA-Web_M10_Z002']
column_filter = ['mass','radius', 'initial_mass', 'initial_z', 'star_age', 'logRho','logT','Teff','energy','photosphere_L', 'photosphere_r', 'star_mass','h1','he3','he4']
column_filter_train = ['mass','radius', 'logRho','logT','energy','h1','he3','he4']  ## considero solo logrho?
n_points=50   # n of points to sample from each profile
r=np.linspace(0,1,n_points)

if dir_names[0] == 'all':
    dir_names = ['MESA-Web_M07_Z00001', 'MESA-Web_M07_Z002', 'MESA-Web_M10_Z002', 'MESA-Web_M10_Z0001',
                 'MESA-Web_M10_Z00001', 'MESA-Web_M15_Z0001', 'MESA-Web_M15_Z00001', 'MESA-Web_M30_Z00001',
                 'MESA-Web_M30_Z002', 'MESA-Web_M50_Z00001', 'MESA-Web_M50_Z002', 'MESA-Web_M50_Z001',
                 'MESA-Web_M5_Z002', 'MESA-Web_M5_Z0001', 'MESA-Web_M1_Z00001', 'MESA-Web_M1_Z0001']

for i,dir_name in enumerate(tqdm(dir_names, desc="Importing data from directories")):

  #print(f"####\t\tIMPORTING DATA FROM FOLDER {dir_name}\t\t####")
  dir_name=os.path.join(cwd,'StellarTracks',dir_name)

  def extract_number(filename):
    match = re.search(r'\d+', filename)  #find the sequence of digits
    return int(match.group()) if match else float('inf')

  filenames=[filename for filename in os.listdir(dir_name) if re.fullmatch('profile[0-9]+\.data',filename)]
  filenames=sorted(filenames, key=extract_number) #sort the elements according to the number in the name

  for j,filename in enumerate(tqdm(filenames, desc=f"Importing from {dir_name}", leave=False)):

    #print(f"####\t\t\tIMPORTING FILE {filename}\t\t\t####")  
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

print('linear train_df shape:', linear_train_df.shape)
#print(linear_train_df)

#for i in range(linear_train_df.shape[1]):
#  plt.plot(r,linear_train_df[f"log_rho_{i}"])
#  plt.show()

x_train, x_test = train_test_split(linear_train_df.T, test_size=0.2)
print ('train shape :', x_train.shape) # (69, 50)
print ('test shape:', x_test.shape) #(18, 50)
#print(x_train)
#print(x_test)


# Reshape the data to match the input shape expected by the model
# The input shape is (batch_size, sequence_length, input_dim)
# In uor case, sequence_length = 50 and input_dim = n_features

x_train_tf = tf.reshape(x_train, ( x_train.shape[0], 50, 1))
x_test_tf = tf.reshape(x_test, ( x_test.shape[0], 50,1))

# Print the shapes to verify
print("x_train_tf shape:", x_train_tf.shape)
print("x_test_tf shape:", x_test_tf.shape)




slope = 0.4 # Best performing ReLU slope found
class Network(tf.keras.Model):
    def __init__(self, hyperparameters):
        super(Network, self).__init__()

        # The hyperparameters of the network are saved for reproduction
        self.hyperparameters = hyperparameters
        
        self.input_size = hyperparameters['input_size']
        #self.hidden_size = hyperparameters['hidden_size']
        self.output_size = hyperparameters['output_size']
        self.activation = hyperparameters['activation']
        self.latent_dim = hyperparameters['latent_dim']
        self.filters = hyperparameters['filters']
        self.padding = hyperparameters['padding']

        if (self.activation == 'relu'):
            act = tf.keras.layers.ReLU()
        if (self.activation == 'elu'):
            act = tf.keras.layers.ELU()
        if (self.activation == 'leakyrelu'):
            act = tf.keras.layers.LeakyReLU(slope)

        # Define the layers of the network ( encoder and decoder)
        
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=self.filters[0], kernel_size=3, strides=2, padding=self.padding[0], activation=act), # 1D 128 kernels of length=36  ##filters -> dimesion of output space
            tf.keras.layers.Conv1D(filters=self.filters[1], kernel_size=3, strides=2, padding=self.padding[1], activation=act),
            tf.keras.layers.Conv1D(filters=self.filters[2], kernel_size=3, strides=2, padding=self.padding[1], activation=act),
            tf.keras.layers.Conv1D(filters=self.latent_dim, kernel_size=3, strides=2, padding=self.padding[1], activation=act), # 1D 4*64 kernels of length=18            
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(filters=self.filters[2], kernel_size=3, strides=2,padding= self.padding[1], activation=act), # Reverse of NiN
            tf.keras.layers.Conv1DTranspose(filters=self.filters[1], kernel_size=3, strides=2,padding= self.padding[1], activation=act),
            tf.keras.layers.Conv1DTranspose(filters=self.filters[0], kernel_size=3, strides=2,padding= self.padding[0], activation=act),
            tf.keras.layers.Conv1DTranspose(filters=self.output_size, kernel_size=3, strides=2, padding= self.padding[1], activation=act), # Reverse of kernel 3
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

    ## Ho copiato il metodo get_param_names() di svm.SVC()
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    
    ## Ho copiato il metodo get_params() di svm.SVC()
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    
filters = np.array([500, 250, 70])

hyperparameters = {
'input_size' : x_train_tf.shape[1:],
'filters' : filters,
'output_size': x_train_tf.shape[-1],
'activation': 'leakyrelu',
'latent_dim': 4,
'padding': np.array(['valid', 'same'])
}


autoencoder = Network(hyperparameters)

# compile
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# training 
history = autoencoder.fit(x_train_tf, x_train_tf,
                        epochs=100,
                        shuffle=True,
                        validation_data=(x_test_tf, x_test_tf),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= 20, verbose = 1)] )

print(f"Latent Dim = {hyperparameters['latent_dim']}:")
print("Loss history:", history.history['loss'])
print("Validation Loss history:", history.history['val_loss'])

autoencoder.encoder.summary()
autoencoder.decoder.summary()

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title(f'Training Loss VS Validation Loss - Latent Dim = {hyperparameters['latent_dim']}')
plt.legend()
plt.show()

x_reconstructed = autoencoder.predict(x_test_tf)
print('Shape of x_recontructed:', x_reconstructed.shape)

# Plot the original and reconstructed data
n = 5
plt.figure(figsize= (18,8))
plt.suptitle(f'Examples of fit with latent dimention = {hyperparameters['latent_dim']}', fontsize= 16)
for j in range(n):
    ax= plt.subplot(2, n, j + 1)
    plt.scatter(r, x_test_tf.numpy()[j].reshape(-1, n_points)[0], color= 'blue', label='Original', s= 7)
    plt.scatter(r, x_reconstructed[j].reshape(-1, n_points)[0], color= 'red', label = 'Reconstructed', s= 7)
    if j == 0:
        plt.legend(loc='best')
    plt.axis('on')
    if j ==0:
        plt.ylabel('Original vs Recostructed')
    # Display the difference
    ax= plt.subplot(2, n, j+1+n)
    difference= x_test_tf.numpy()[j].reshape(-1, n_points)[0] - x_reconstructed[j].reshape(-1, n_points)[0]       
    mean_difference = np.mean(difference)
    plt.scatter(r, difference, color='green', s=7, label = 'Difference')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axhline(mean_difference, color ='darkred', linestyle='--', linewidth = 1.5, label ='Mean difference')

    if j == 0:
        plt.ylabel('Difference')
        plt.legend(loc= 'best')
    plt.axis("on")

plt.show()

################## GRID SEARCH ########################

hyperparameters = {'latent_dim':(3,4,5,6), 'padding':('valid', 'same')}

clf = GridSearchCV(autoencoder, hyperparameters, scoring = make_scorer(mean_squared_error))
clf.fit(x_train_tf, None)
clf
    