# Imports
import numpy as np
from numpy import argmax, argmin
from scipy.stats import norm
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import time
import pandas as pd
import os

# Settings
np.set_printoptions(suppress=True)

# Variables
dims = 4

# Parameter Space
param_bounds = [
    np.linspace(start=0, stop=5, num=21),
    np.linspace(start=10, stop=100, num=46),
    np.linspace(start=0, stop=5, num=21),
    np.linspace(start=100, stop=300, num=101)
]
param_space = np.array( np.meshgrid(param_bounds[0],
                                    param_bounds[1],
                                    param_bounds[2],
                                    param_bounds[3])).T.reshape(-1,dims)

csv_path = os.path.abspath(os.path.join(
    os.getcwd(),
    os.pardir,
    'datasets',
    'quality_scores.csv'
))
df = pd.read_csv(csv_path)



# Standardize data
def standardize(x):
    return (x - np.mean(x,axis=0)) / np.std(x,axis=0)+1e-9

# GP model predictions
def surrogate(model, x):
    x = standardize(x)
    return model.predict(x ,return_std=True)

# Expected Improvement (EI)
def acquisition(x_sample, param_space, model):
    Xi = 0.1
    yhat, _ = surrogate(model, x_sample)
    best = max(-yhat)
    mu, std = surrogate(model, param_space)
    imp = (mu - best - Xi)
    with np.errstate(divide='warn'):
        z = imp / std
        probs = imp *  norm.cdf( z ) + std * norm.pdf( z )
        probs[std==0.0] = 0.0
    return probs

# Optimal parameters using acq. function
def opt_acquisition(x_sample, param_space, model):
    scores = acquisition(x_sample,param_space, model)
    ix = argmax(-scores)
    return param_space[ix], ix


### Let's get it started ###
# Initialize model
model = GaussianProcessRegressor() # kernel=kernel, n_restarts_optimizer=1

# Define init data
x_init = np.array(df[["Param1", "Param2", "Param4", "Param5"]].iloc[:1000,:])
y_init = np.array(df["Area sum"].iloc[:1000])

# Fit model on init data
model.fit(x_init, y_init)

# Apply GP on parameter space 
ysamples, std = surrogate(model, param_space)

# Calculate best point to sample
best_x, ix = opt_acquisition(x_init,param_space,model)
print(f"Best point: {best_x}")

# Calculate estimate and deviation (mean and std)
est_mean, est_std = surrogate(model, [best_x])
print(f"Estimated mean: {-int(est_mean)} with deviation: {est_std}")