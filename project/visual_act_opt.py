import os
import matplotlib.pyplot as plt
import pandas as pd
import ast

id=3
plot_type='activation'  #activation or optimizer
title='Final Loss of 400-100-50-7-50-100-400 networks \n with fixed adamax optimizer Vs chosen Activations'

activations=['relu','leaky_relu','gelu','softplus','elu','selu','silu']
optimizers=['nadam','adam','rmsprop','sgd','adagrad','adadelta','ftrl','adamax','adamw','lion']

if plot_type=='activation':
    filenames=[f'results{id}-{act}.csv' for act in activations]
elif plot_type=='optimizer':
    filenames=[f'results{id}-{opt}.csv' for opt in optimizers]

fig = plt.figure(figsize=(10,5))

for file in filenames:
    data=pd.read_csv(file)
    data['avg_final_val_loss'] = data['avg_final_val_loss'].apply(lambda x: ast.literal_eval(x))
    data['avg_final_val_loss'] = data['avg_final_val_loss'].apply(lambda x: pd.Series(x).mean())
    plt.scatter(data[plot_type],data['avg_final_val_loss'])
    
plt.title(title)
plt.xlabel(plot_type)
plt.ylabel('Final Loss (mean over last 15 steps)')
plt.savefig(f'loss_Vs_{plot_type}_{id}.png')
plt.show()

