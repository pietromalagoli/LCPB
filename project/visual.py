import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

#import and show data
performance=pd.read_csv('results_leaky_relu.csv')
print(performance.head())
print(performance.columns)

#transform string into series because reading from file gives problems
performance['avg_final_val_loss'] = performance['avg_final_val_loss'].apply(lambda x: ast.literal_eval(x))

#create new column with average final losses
performance['avg_loss']=performance['avg_final_val_loss'].apply(lambda x: pd.Series(x).mean())

#determine best model
best=performance['avg_loss'].idxmin()
print(best)
print(performance.iloc[best])

#plot
fig, ax = plt.subplots(figsize = (9, 6))
ax.scatter(x=performance['hidden_neurons'],y=performance['avg_loss'])
ax.set_yscale('log')
plt.show()