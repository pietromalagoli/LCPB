# Libraries
import aux
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import re

from sklearn.model_selection import train_test_split
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm 

###     DA FARE!!
#       - aggiungi trainloss plots al codice comune



#### PREPROCESSING

x_train_tf, x_test_tf, features, r_points = aux.preprocess(dir_names=['all'], features=['mass', 'logRho', 'logT', 'energy'], n_points=50)


### hYPERPARAMETERS SETUP

hyperparameters = {
    'input_size': x_train_tf.shape[1:],  # Input shape
    'output_size': x_train_tf.shape[-1],  # Output shape
    'activation': 'leakyrelu',  # Activation function
    'latent_dim': 4  # Latent dimension
}

latent_dim_mse = []  # List to store MSE for each latent dimension

# Loop over latent dimensions (3 to 5)
for latent_dim in range(4, 5):
    print(f"\nTraining with latent dim = {latent_dim}")
    hyperparameters['latent_dim'] = latent_dim
    autoencoder = aux.Network(hyperparameters)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    # Train the model
    history = autoencoder.fit(x_train_tf, x_train_tf,
                              epochs=100,
                              shuffle=True,
                              validation_data=(x_test_tf, x_test_tf),
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])
    
    print(f"Latent Dim = {latent_dim}:")
    print("Loss history:", history.history['loss'])
    print("Validation Loss history:", history.history['val_loss'])

    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    # Make predictions on the test data
    x_reconstructed = autoencoder.predict(x_test_tf)
    print('Shape of x_reconstructed:', x_reconstructed.shape)

    ### Calculate MSE (mean squared error) for this latent dimension
    mse = np.mean((x_test_tf.numpy() - x_reconstructed) ** 2)
    latent_dim_mse.append((latent_dim, mse))  # Save the latent dimension and its corresponding MSE


    # Plot the original and reconstructed data
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns

    # Plot original vs reconstructed for each feature in the first row
    for i, feature in enumerate(features):
        ax = axes[0, i]
        ax.scatter(r_points, x_test_tf[0, :, i], label='Original', color='blue', marker='o')
        ax.scatter(r_points, x_reconstructed[0, :, i], label='Reconstructed', color='red', marker='x')
        ax.set_title(f'{feature} (Latent Dim: {latent_dim})')
        ax.set_xlabel('Normalized Radius')
        ax.set_ylabel(f'{feature}')
        ax.grid(True)
        if i == 0:
            ax.legend(loc='best')

    # Plot the difference between original and reconstructed for each feature in the second row
    for i, feature in enumerate(features):
        ax = axes[1, i]
        difference = np.abs(x_test_tf[0, :, i] - x_reconstructed[0, :, i])
        mean_difference = np.mean(difference)
        ax.scatter(r_points, difference, label=f'{feature} Difference', color='darkgreen', marker='o')
        ax.axhline(0, color='black', linewidth=0.7)  # Horizontal line at y=0
        ax.axhline(mean_difference, color ='darkred', linestyle='--', linewidth = 1.5, label ='Mean difference')
        ax.set_title(f'{feature} Difference (Latent Dim: {latent_dim})')
        ax.set_xlabel('Normalized Radius')
        ax.set_ylabel(f'{feature} Difference')
        ax.grid(True) 
        ax.set_xlim(0, 1)  
        ax.set_ylim(0, 2)
    
    # Adjust layout
    plt.tight_layout()

    file_save_dir = os.path.join(os.getcwd(), "Graphs", f"LatentDim_{latent_dim}_Comparison.png")
    plt.savefig(file_save_dir)
    plt.close()

    for latent_dim, mse in latent_dim_mse:
        print(f"Latent dimension: {latent_dim}, Reconstruction MSE: {mse:.6f}")

# Plot MSE vs Latent Dimension and save it
plt.figure(figsize=(8, 6))
latent_dims, mse_values = zip(*latent_dim_mse)
plt.plot(latent_dims, mse_values, marker='o')
plt.title('Reconstruction Error (MSE) vs Latent Dimension')
plt.xlabel('Latent Dimension')
plt.ylabel('Reconstruction MSE')
plt.grid(True)

# Save the MSE graph to the same folder
file_save_dir = os.path.join(os.getcwd(), "Graphs", "MSE_vs_LatentDimension.png")
plt.savefig(file_save_dir)
plt.show()
plt.close()

print(features)