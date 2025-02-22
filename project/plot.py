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

plt.rcParams.update({'font.size': 25})


##########
# PARAMETERS OPTIMIZATION

save_dir = os.path.join(os.getcwd(), "Graphs")  

optimizers=['adam','nadam','rmsprop','adagrad','adadelta','ftrl','adamax','lion']
activations=['leaky_relu','relu','gelu','softplus','elu','selu','silu']

# activations (fixed adam)
performance_data=pd.DataFrame()

optimizer = optimizers[0]

# Hyperparameters setup
hyperparameters = {
    'latent_dim': 1  # Fixed latent dimension
}

plot_type='activation'  #activation or optimizer
title=f'Activations performances with fixed optimizer as {optimizer}'

file=f'results-{optimizer}-{hyperparameters['latent_dim']}.csv'

fig = plt.figure(figsize=(15,7.5))

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

plot_type='optimizer'  #activation or optimizer
title=f'Optimizers performances with fixed optimizer as {activation}'

file=f'results-{activation}-{hyperparameters['latent_dim']}.csv'

fig = plt.figure(figsize=(15,7.5))

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