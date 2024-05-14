import mesa_web as mw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

filename='StellarTracks/MESA-Web_M07_Z00001'

profile_dfs=[]

for file_name in os.listdir(filename):
  if file_name.endswith('.data'):
    file_path= os.path.join(filename, file_name)
    data=mw.read_profile(file_path)

    profile_df = pd.DataFrame(data)
    profile_dfs.append(profile_df)

all_profiles_df= pd.concat(profile_dfs, ignore_index= True)



#data=mw.read_profile(filename)

#profile1_df=pd.DataFrame(data)
column_filter = ['mass','radius', 'initial_mass', 'initial_z', 'star_age', 'logRho','logT','Teff','energy','photosphere_L', 'photosphere_r', 'star_mass','h1','he3','he4']
column_filter_train = ['mass','radius', 'logRho','logT','energy','h1','he3','he4']


# Create a new DataFrame with only the selected columns
#filtered_profile1_df = profile1_df[column_filter].copy()
filtered_profiles_df= all_profiles_df[column_filter].copy()
print(filtered_profiles_df.shape[0]) 

# Create a new DataFrame with only the selected columns for autoencoder training
#train_filtered_profile1_df = profile1_df[column_filter_train].copy()
train_filtered_profiles_df = all_profiles_df[column_filter_train].copy()

#Normalization process#
tot_radius_1=filtered_profiles_df['photosphere_r']
tot_mass=filtered_profiles_df['star_mass']
#From log to linear scale
train_filtered_profiles_df['logRho']=10**(train_filtered_profiles_df['logRho'])
train_filtered_profiles_df['logT']=10**(train_filtered_profiles_df['logT'])

norm_radius=(filtered_profiles_df['radius']-filtered_profiles_df['radius'].min())/(tot_radius_1-filtered_profiles_df['radius'].min())
norm_mass=(filtered_profiles_df['mass']-filtered_profiles_df['mass'].min())/(tot_mass-filtered_profiles_df['mass'].min())
norm_rho=(train_filtered_profiles_df['logRho']-train_filtered_profiles_df['logRho'].min())/(train_filtered_profiles_df['logRho'].max()-train_filtered_profiles_df['logRho'].min())
norm_t=(train_filtered_profiles_df['logT']-train_filtered_profiles_df['logT'].min())/(train_filtered_profiles_df['logT'].max()-train_filtered_profiles_df['logT'].min())
norm_energy=(train_filtered_profiles_df['energy']-train_filtered_profiles_df['energy'].min())/(train_filtered_profiles_df['energy'].max()-train_filtered_profiles_df['energy'].min())

norm_radius=np.asarray(norm_radius)
norm_mass=np.asarray(norm_mass)
norm_rho=np.asarray(norm_rho)
norm_t=np.asarray(norm_t)
norm_energy=np.asarray(norm_energy)


linear_train_df=pd.DataFrame([norm_radius,norm_mass,norm_rho,norm_t,norm_energy])

#linear_train_df=linear_train_df.T
print(linear_train_df.T)

plt.plot(norm_radius,filtered_profiles_df['logRho'])
plt.show()

plt.plot(norm_radius,filtered_profiles_df['logT'])
plt.show()

x_train, x_test = train_test_split(linear_train_df.T[2], test_size=0.2, random_state=2404)
print (x_train.shape)
print (x_test.shape)

for i in range(1,3,1):
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
        layers.Dense(394, activation='sigmoid'),
        layers.Reshape((394, 1))
      ])

    def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded

  autoencoder = Autoencoder(latent_dim)

  autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

  history = autoencoder.fit(x_train, x_train,
                  epochs=10,
                  shuffle=True,
                  validation_data=(x_test, x_test))

  autoencoder.encoder.summary()

  autoencoder.decoder.summary()

  plt.plot(history.history["loss"], label="Training Loss")
  plt.plot(history.history["val_loss"], label="Validation Loss")
  plt.title(f'Training Loss VS Validation Loss - Latent Dim = %d' % i)
  plt.legend()
  plt.show()
