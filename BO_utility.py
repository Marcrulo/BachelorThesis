import numpy as np
from numpy import argmax, argmin
from scipy.stats import norm


# Standardize data
def standardize(x, transform_me):
    #try:
    return (transform_me - np.mean(x,axis=0)) / (np.std(x,axis=0)+1e-9)
    #except Exception as e:
    #    print(e)
    #    print(x)
    #    print(x.shape)
    #    print(type(x))
    #    assert "Division error"

# Reverse Standardize data
def unstandardize(x, standardized):
    std = np.std(x,axis=0)+1e-9
    mean = np.mean(x,axis=0)
    old = standardized*std + mean
    return old






# GP model predictions
def surrogate(model, x):
    return model.predict(x ,return_std=True)


# Expected Improvement (EI)
# def acquisition(x_sample, param_space, model):
#     Xi = 0
#     yhat, _ = surrogate(model, x_sample)
#     best = max(yhat)
#     mu, std = surrogate(model, param_space)
#     imp = (mu - best - Xi)
#     with np.errstate(divide='warn'):
#         z = imp / std
#         probs = imp *  norm.cdf( z ) + std * norm.pdf( z )
#         probs[std==0.0] = 0.0
#     return probs 

# Probability of Improvement (PI)
# def acquisition(x_sample, param_space, model):
#     yhat, _ = surrogate(model, x_sample)
#     best = max(yhat)
#     mu, std = surrogate(model, param_space)
#     imp = (mu - best)
#     with np.errstate(divide='warn'):
#         z = imp / std
#         probs = norm.cdf( z )
#         probs[std==0.0] = 0.0
#     return probs 


# Upper Confidence Bounds (UCB)
def acquisition(x_sample, param_space, model):
    kappa = 1
    mu, std = surrogate(model, param_space)
    probs = mu + kappa * std
    return probs

#def acquisition(x_sample, param_space, model):
#    mu, std = surrogate(model, param_space)
#    return std

# Lower Confidence Bounds (LCB)
# def acquisition(x_sample, param_space, model):
#     kappa = 1
#     mu, std = surrogate(model, param_space)
#     probs = mu - kappa * std
#     return probs

# Optimal parameters using acq. function
def opt_acquisition(x_sample, param_space, model):
    scores = acquisition(x_sample,param_space, model)
    while True:
        ix = argmax(scores)
        if not any(np.array_equal(x, param_space[ix]) for x in x_sample):
            break
        scores[ix] = -10000000
    return param_space[ix], ix