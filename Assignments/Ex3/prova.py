import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# AdaBoost Algorithm
from sklearn.ensemble import AdaBoostClassifier
# Gradient Boosting 
from sklearn.ensemble import GradientBoostingClassifier
# XGBoost 
import xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance, to_graphviz, plot_tree
print("XGBoost version:",xgboost.__version__)
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, cross_val_score
import os

mycmap = "winter"
mpl.rcParams['image.cmap'] = mycmap
plt.rcParams['font.size'] = 13
np.random.seed(12345)

path=os.getcwd()
dname="\Assignments\Ex3\DATA\\"
str0="_XGB_24.dat"
fnamex=path+dname+'x'+str0
fnamey=path+dname+'y'+str0
x = np.loadtxt(fnamex, delimiter=" ",dtype=float)
y = np.loadtxt(fnamey)
y = y.astype(int)
N,L = len(x), len(x[0])
num_neurons=20
num_layers=2
activation_function='relu'
perc=0.7
n_rep=30
def build_nn(num_layers,num_neurons,activation_function):
    input_shape = x.shape[1]
    model_nn = Sequential()
    model_nn.add(Dense(units=num_neurons, activation=activation_function, input_shape=(input_shape,)))
    for i in range(num_layers-1):
        model_nn.add(Dense(units=num_neurons, activation=activation_function))
    model_nn.add(Dense(units=2,activation='softmax'))
    opt = keras.optimizers.Adam()
    model_nn.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    return model_nn

num_layers=[2,3,4,5,6,7,8,9,10]                      #num of nn layers(input and output excluded)
num_neurons=[5,10,15,20,25,30,35,40]                        #num of neurons per layer
activation_functions=['relu']         #activation function for the hidden layers
max_acc=0
max_desc=""
perc=0.7
n_rep=20
num_indices=int(perc*len(x))        #get quantity of data points in training
data_indices = np.arange(len(x))    #get indices
mean_acc_list=[]

for nl in num_layers:
    for nn in num_neurons:
        for act in activation_functions:
            accuracies_nn=[]
            print(f"--RUNNING--n_lay: {nl}, n_neu: {nn}, act: {act}")
            for i in range(n_rep):
                np.random.shuffle(data_indices)     #shuffle indices
                train_indices = data_indices[:num_indices]  #get training set indices
                test_indices = data_indices[num_indices:]   #get test set indices
                
                x_train, y_train = x[train_indices], y[train_indices]
                x_test, y_test = x[test_indices], y[test_indices]
                y_train = keras.utils.to_categorical(y_train, num_classes=2)
                y_test = keras.utils.to_categorical(y_test, num_classes=2)
                
                model=build_nn(num_layers=nl,num_neurons=nn,activation_function=act)
                model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=0)
                accuracy_nn = model.evaluate(x_test, y_test, verbose=0)[1]  #accuracy is at index 1
                accuracies_nn.append(accuracy_nn)
                del model,x_train,x_test,y_train,y_test,train_indices,test_indices      #just to free memory
            mean_acc=np.mean(accuracies_nn)
            mean_acc_list.append(mean_acc)
            print(mean_acc)
            if(mean_acc>max_acc):
                max_acc = mean_acc
                max_desc = f"n_lay: {nl}, n_neu: {nn}, act: {act}"
print(f"Best mean accuracy ({max_acc}) obtained by the model with parameters: {max_desc}")