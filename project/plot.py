import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

optimizers=['adam','nadam','rmsprop','adagrad','adadelta','ftrl','adamax','adamw','lion']
activations=['relu','leaky_relu','gelu','softplus','elu','selu','silu']

activation = activations[0]

plot_type='optimizer'  #activation or optimizer
title=f'Optimizers performances with fixed optimizer as {activation}'

file=f'results-{activation}.csv'

fig = plt.figure(figsize=(10,5))

data=pd.read_csv(file)

colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))
for i, (x, y) in enumerate(zip(data[plot_type], data['avg_final_val_loss'])):
    plt.scatter(x, y, color=colors[i])
    
plt.title(title)
plt.xlabel(plot_type)
plt.ylabel('Final Loss (mean over last 15 steps)')
plt.savefig('opt_opt1.png')