import mesa_web as mw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 
import re

from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras import layers, losses, metrics
from keras.datasets import fashion_mnist
from keras.models import Model
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler

##Autoencoder class definition
class Autoencoder(Model):
    
    def __init__(self, encoder_neurons, decoder_neurons, encoder_activations, decoder_activations):

      ##input check
      if not len(encoder_neurons)==len(encoder_activations):
        raise ValueError('The vector of neuron numbers for the encoder should be the same size of the activations')
      if not len(decoder_neurons)==len(decoder_activations):
        raise ValueError('The vector of neuron numbers for the decoder should be the same size of the activations')

      encoder_layers=[]
      decoder_layers=[]

      ##define the encoder
      input_shape = keras.Input(shape=(decoder_neurons[-1],))
      encoded = layers.Dense(encoder_neurons[0],activation=encoder_activations[0])(input_shape)
      for i in range(1,len(encoder_neurons)):
        encoded = layers.Dense(encoder_neurons[i],activation=encoder_activations[i])(encoded)

      ##define the decoder
      decoded = layers.Dense(decoder_neurons[0],activation=decoder_activations[0])(encoded)
      for i in range(1,len(decoder_neurons)):
        decoded = layers.Dense(decoder_neurons[i],activation=decoder_activations[i])(decoded)
      
      super().__init__(input_shape,decoded)

###Import parameters
cwd=os.getcwd()
dir_names = ['all']    #type 'all' if you want to use all the data  
column_filter = ['mass','radius', 'initial_mass', 'initial_z', 'star_age', 'logRho','logT','Teff','energy','photosphere_L', 'photosphere_r', 'star_mass','h1','he3','he4']
column_filter_train = ['mass','radius', 'logRho','logT','energy','h1','he3','he4']
n_points=100
r=np.linspace(0,1,n_points+1)[1:]

if (dir_names[0]=='all'):
  dir_names=['MESA-Web_M07_Z00001', 'MESA-Web_M07_Z002','MESA-Web_M10_Z002','MESA-Web_M10_Z0001', 'MESA-Web_M10_Z00001','MESA-Web_M15_Z0001', 'MESA-Web_M15_Z00001', 'MESA-Web_M30_Z00001', 'MESA-Web_M30_Z002','MESA-Web_M50_Z00001', 'MESA-Web_M50_Z002', 'MESA-Web_M50_Z001', 'MESA-Web_M5_Z002', 'MESA-Web_M5_Z0001', 'MESA-Web_M1_Z00001', 'MESA-Web_M1_Z0001']

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


scaler = MinMaxScaler()
linear_train_df = pd.DataFrame(scaler.fit_transform(linear_train_df), columns=linear_train_df.columns)

print(linear_train_df.shape)
print(linear_train_df)


x_train, x_test = train_test_split(linear_train_df.T, test_size=0.2)
x_train = x_train.values
x_test = x_test.values
print(x_train.shape)
print(x_test.shape)
print(x_train)
print(x_test)
shape = x_test.shape[1:]
save_graphs=True

for i in range(1,10):

  ###Autoencoder parameters
  encoder_neurons=[100,i]                                         #last value should be the latent dimention
  decoder_neurons=encoder_neurons[:len(encoder_neurons)-1][::-1]            #same inverted values for the hidden layers
  decoder_neurons.append(n_points)                                          #last value should be original size
  encoder_activations=['relu']*len(encoder_neurons)                         #with relu better learning
  decoder_activations=['relu']*(len(encoder_neurons)-1)                     #relu for better learning
  decoder_activations.append('sigmoid')                                     #at least the last value should be sigmoid

  
  autoencoder = Autoencoder(encoder_neurons=encoder_neurons,decoder_neurons=decoder_neurons,\
                            encoder_activations=encoder_activations,decoder_activations=decoder_activations)

  autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

  autoencoder.summary()
  #for layer in autoencoder.layers: print(layer.get_config(), layer.get_weights())
  
  history = autoencoder.fit(x_train, x_train,
                  epochs=1000,
                  shuffle=True,
                  validation_data=(x_test, x_test),
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
  
  #print(history.history)

  if save_graphs:
    file_save_dir=os.path.join(os.getcwd(),"Graphs",f"TrainValLoss_dim_{i}.png")
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f'Training Loss VS Validation Loss - Latent Dim = {i}')
    plt.legend()
    plt.savefig(file_save_dir)
    plt.close()
  
  x_reconstructed = autoencoder.predict(x_test)
  print(x_reconstructed.shape)
  
  n = 10  # How many digits we will display
  plt.figure(figsize=(20, 4))
  plt.axis("off")
  plt.title(f'Examples of fit with latent dimension = {i}')
  for j in range(n):
      # Display original
      ax = plt.subplot(2, n, j + 1)
      plt.scatter(r,x_test[j])
      plt.gray()

      # Display reconstruction
      ax = plt.subplot(2, n, j + 1 + n)
      plt.scatter(r,x_reconstructed[j])
      plt.gray()
  
  file_save_dir=os.path.join(os.getcwd(),"Graphs",f"OriginalReconstructed_{i}.png")
  plt.savefig(file_save_dir)
  plt.close()