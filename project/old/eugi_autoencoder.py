# Libraries
import mesa_web as mw
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import re

from sklearn.model_selection import train_test_split
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm 


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
            """
            column_values = filtered_profile_df[column].values.reshape(-1, 1)  # Reshape for sklearn
            scalers[column].fit(column_values)  # Fit scaler
            norm = scalers[column].transform(column_values).flatten()  # Normalize data
            """
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
print("Final length of all profiles",len(all_profiles))
print("Final shape of all profiles:", all_profiles.shape)

# Split the data into training and testing sets
x_train, x_test = train_test_split(all_profiles, test_size=0.2, shuffle=False)
print ('train shape :', x_train.shape)   # (train_samples, n_points, num_features)
print ('test shape:', x_test.shape)      # (train_samples, n_points, num_features)

# Reshape the data to match the input shape expected by the model
num_features = len(column_filter_train)

x_train_tf = tf.reshape(x_train, ( x_train.shape[0], n_points, num_features))
x_test_tf = tf.reshape(x_test, ( x_test.shape[0], n_points, num_features))

# Print the shapes to verify
print("x_train_tf shape:", x_train_tf.shape) #(1148, 50, 4)
print("x_test_tf shape:", x_test_tf.shape)   #(288, 50, 4)



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

        #if self.activation == 'relu':
            #act = tf.keras.layers.ReLU()
        #if self.activation == 'elu':
            #act = tf.keras.layers.ELU()
        #if self.activation == 'leakyrelu':
        act = tf.keras.layers.LeakyReLU(alpha= 0.35)

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
        # Pass the input through the encoder to get the encoded representation  
        encoded = self.encoder(x)
        # Pass the encoded representation through the decoder to reconstruct the input
        decoded = self.decoder(encoded)

        # Return the reconstructed output
        return decoded


# Hyperparameters setup
hyperparameters = {
    'input_size': x_train_tf.shape[1:],  # Input shape
    'output_size': x_train_tf.shape[-1],  # Output shape
    'activation': 'leakyrelu',  # Fixed activation function
    'latent_dim': 4  # Fixed latent dimension
}


'''
#activation_functions = ['relu', 'elu', 'leakyrelu']  # Activation functions to test
#activation_mse = []  # List to store MSE for each activation function
alpha_values = np.linspace(0.01, 1, 10)  # Range of alpha values to test (e.g., 0.01, 0.1, ..., 1)
alpha_mse = []  # List to store MSE for each alpha value

# Loop 

for alpha in alpha_values:
    print(f"\nTraining with latent dim = {alpha}")
    hyperparameters['activation'] = 'leakyrelu'
'''
autoencoder = Network(hyperparameters)

# Compile the model
autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

# Train the model
history = autoencoder.fit(x_train_tf, x_train_tf,
                            epochs=100,
                            shuffle=True,
                            validation_data=(x_test_tf, x_test_tf),
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])
    
#print(f"Latent Dim = {activation}:")
print("Loss history:", history.history['loss'])
print("Validation Loss history:", history.history['val_loss'])

# Display model summary
autoencoder.encoder.summary()
autoencoder.decoder.summary()

# Make predictions on the test data
x_reconstructed = autoencoder.predict(x_test_tf)
print('Shape of x_reconstructed:', x_reconstructed.shape)

### Calculate MSE (mean squared error) for this latent dimension
mse = np.mean((x_test_tf.numpy() - x_reconstructed) ** 2)
#activation_mse.append((activation, mse))  # Save the latent dimension and its corresponding MSE
print(f"Reconstruction MSE: {mse:.6f}")

# Plot the original and reconstructed data
fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns

# Plot original vs reconstructed for each feature in the first row
features = ['mass', 'logRho', 'logT', 'logEnergy']
for i, feature in enumerate(features):
    ax = axes[0, i]
    ax.scatter(r, x_test_tf[0, :, i], label='Original', color='blue', marker='o')
    ax.scatter(r, x_reconstructed[0, :, i], label='Reconstructed', color='red', marker='x')
    ax.set_title(f'{feature} vs Normalized Radius')
    ax.set_xlabel('Normalized Radius')
    ax.set_ylabel(f'{feature}')
    ax.grid(True)
    ax.set_xlim(0, 1)  
    if i == 0:
        ax.legend(loc='best')

# Plot the difference between original and reconstructed for each feature in the second row
for i, feature in enumerate(features):
    ax = axes[1, i]
    difference = np.abs(x_test_tf[0, :, i] - x_reconstructed[0, :, i])
    mean_difference = np.mean(difference)
    ax.scatter(r, difference, label=f'{feature} Difference', color='darkgreen', marker='o')
    ax.axhline(0, color='black', linewidth=0.7)  # Horizontal line at y=0
    ax.axhline(mean_difference, color ='darkred', linestyle='--', linewidth = 1.5, label ='Mean difference')
    ax.set_title(f'{feature} Difference')
    ax.set_xlabel('Normalized Radius')
    ax.set_ylabel(f'{feature} Difference')
    ax.grid(True) 
    ax.set_xlim(0, 1)  
    if feature == 'logEnergy':
        ax.set_ylim(0, 1.9)
    else:
        ax.set_ylim(0, 1)



    
# Adjust layout
plt.tight_layout()

file_save_dir = os.path.join(os.getcwd(), "Graphs", "Original_vs_Reconstructed_Comparison.png")
plt.savefig(file_save_dir)
plt.close()


#######################################################################################
##################### mse  per ogni features ##########################################
#######################################################################################

from sklearn.metrics import mean_squared_error
def calculate_mse(x_test, x_reconstructed, features):
    mse_values = []
    for i, feature in enumerate(features):
        mse = mean_squared_error(x_test[0, :, i], x_reconstructed[0, :, i])
        mse_values.append(mse)
    return mse_values

def plot_mse(mse_values, features):
    # Plotting the MSE for each feature
    plt.figure(figsize=(10, 6))
    plt.bar(features, mse_values, color='lightblue')
    plt.xlabel('Features')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE per Feature')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Calcola l'MSE per ogni feature
mse_values = calculate_mse(x_test_tf, x_reconstructed, features)

# Mostra il grafico con gli errori per ogni feature
plot_mse(mse_values, features)




###### codice vecchio non guardare
'''
# Hyperparameters setup
hyperparameters = {
    'input_size': x_train_tf.shape[1:],  # Input shape
    'output_size': x_train_tf.shape[-1],  # Output shape
    'activation': 'leakyrelu',  # Fixed activation function
    'latent_dim': 4  # Fixed Latent dimension
}
'''
'''
alpha_values = np.linspace(0.01, 1, 10)  # Range of alpha values to test (e.g., 0.01, 0.1, ..., 1)
alpha_mse = []  # List to store MSE for each alpha

# Loop over alpha values for LeakyReLU
for alpha in alpha_values:
    print(f"\nTraining with LeakyReLU alpha = {alpha}")
    
    # Update the hyperparameters with the specific alpha
    hyperparameters['activation'] = 'leakyrelu'
    
    # Define the network with the specific alpha for LeakyReLU
    class NetworkWithAlpha(tf.keras.Model):
        def __init__(self, hyperparameters, alpha):
            super(NetworkWithAlpha, self).__init__()
            self.hyperparameters = hyperparameters
            self.input_size = hyperparameters['input_size']
            self.output_size = hyperparameters['output_size']
            self.latent_dim = hyperparameters['latent_dim']

            # LeakyReLU with specific alpha
            act = tf.keras.layers.LeakyReLU(alpha=alpha)

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
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    # Instantiate the model
    autoencoder = NetworkWithAlpha(hyperparameters, alpha)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    # Train the model
    history = autoencoder.fit(x_train_tf, x_train_tf,
                              epochs=100,
                              shuffle=True,
                              validation_data=(x_test_tf, x_test_tf),
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])
    
    print(f"LeakyReLU Alpha = {alpha}:")
    print("Loss history:", history.history['loss'])
    print("Validation Loss history:", history.history['val_loss'])

    # Make predictions on the test data
    x_reconstructed = autoencoder.predict(x_test_tf)
    print('Shape of x_reconstructed:', x_reconstructed.shape)

    # Calculate MSE (mean squared error) for this alpha value
    mse = np.mean((x_test_tf.numpy() - x_reconstructed) ** 2)
    alpha_mse.append((alpha, mse))  # Save the alpha value and its corresponding MSE

    # Optional: Plot the original and reconstructed data for a specific alpha
    if alpha in [0.01, 0.5, 1.0]:  # Example: plot for specific alpha values
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns
        column_filter_train = ['mass', 'logRho', 'logT', 'logEnergy']
        for i, feature in enumerate(column_filter_train):
            ax = axes[0, i]
            ax.scatter(r, x_test_tf[0, :, i], label='Original', color='blue', marker='o')
            ax.scatter(r, x_reconstructed[0, :, i], label='Reconstructed', color='red', marker='x')
            ax.set_title(f'{feature} (Alpha: {alpha})')
            ax.set_xlabel('Normalized Radius')
            ax.set_ylabel(f'{feature}')
            ax.grid(True)
            if i == 0:
                ax.legend(loc='best')

# Print MSE results for each alpha
print("\nMSE Results for each alpha:")
for alpha, mse in alpha_mse:
    print(f"Alpha: {alpha}, MSE: {mse}")

# Plot MSE vs Alpha
alphas, mses = zip(*alpha_mse)
plt.figure(figsize=(10, 6))
plt.plot(alphas, mses, marker='o')
plt.title("MSE vs LeakyReLU Alpha")
plt.xlabel("LeakyReLU Alpha")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.show()
''' 