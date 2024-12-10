import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

#import and show data
filename='results6.csv'
performance=pd.read_csv(filename)
#print(performance.head())
#print(performance.columns)

#transform string into series because reading from file gives problems
performance['avg_final_val_loss'] = performance['avg_final_val_loss'].apply(lambda x: ast.literal_eval(x))
performance['encoder_neurons'] = performance['encoder_neurons'].apply(lambda x: ast.literal_eval(x))

#create new column with average final losses
performance['avg_loss']=performance['avg_final_val_loss'].apply(lambda x: pd.Series(x).mean())
performance['num_layers']=performance['encoder_neurons'].apply(lambda x: (len(x)*2)+3)

#determine best model
best=performance['avg_loss'].idxmin()
print(best)
print(performance.iloc[best])

#plot
fig, ax = plt.subplots(figsize = (9, 6))
ax.scatter(x=performance['hidden_neurons'],y=performance['avg_loss'],s=10)
ax.set_xlabel("Hidden neurons")
ax.set_ylabel("Average final loss (avg of last 15 iterations)")
ax.set_yscale('log')
ax.set_title('Average final loss Vs Number of Hidden Neurons')
fig.savefig(f'run{filename.split('.')[0][-1]}_loss_hn.png')
plt.show()

fig, ax = plt.subplots(figsize = (9, 6))
ax.scatter(x=performance['num_layers'],y=performance['avg_loss'],s=10)
ax.set_yscale('log')
ax.set_xlabel("Number of layers")
ax.set_ylabel("Average final loss (avg of last 15 iterations)")
ax.set_title('Average final loss Vs Total number of network layers')
fig.savefig(f'run{filename.split('.')[0][-1]}_loss_layers.png')
plt.show()

fig, ax = plt.subplots(figsize = (9, 6))
ax.scatter(x=performance['optimizer'],y=performance['avg_loss'],s=10)
ax.set_yscale('log')
ax.set_xlabel("Optimizer")
ax.set_ylabel("Average final loss (avg of last 15 iterations)")
ax.set_title('Average final loss Vs Used Optimizer')
fig.savefig(f'run{filename.split('.')[0][-1]}_loss_opt.png')
plt.show()

print(performance.sort_values(by='avg_loss')[['hidden_neurons','optimizer','avg_loss','encoder_neurons']].head(15)) #