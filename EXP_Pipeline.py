
# Define experiment
exp = input('## Name of experiment: ')


# Imports
from Image_datasets import move_to_datasets
from Image_preprocessing import preprocess
from Image_segmentation import load_processed_images, binarize, BLOB
from BO_utility import standardize, unstandardize, surrogate, acquisition, opt_acquisition

import csv
import os
import re
import numpy as np
from skimage.morphology import disk, dilation, erosion, closing, opening, remove_small_objects
from skimage import io
from tqdm import tqdm
import pandas as pd
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RationalQuadratic, RBF
from sklearn.gaussian_process import GaussianProcessRegressor



"""

Load Image Files

"""

#exps = ["Exp0_20feb", "Exp1_12apr", "Exp2_19apr", "Exp3_25apr", "Exp4_26apr", "Exp5_10may"]
#for exp in exps:    
print("Processing experiment", exp, "...")

# Copy experiment images to the 'raw' folder
exp_path = os.path.abspath(os.path.join(
        os.getcwd(),
        os.pardir,
        'experiments2',
        exp,
    ))
if os.path.exists(exp_path):
    # Check if parameters are defined
    #while True:
    #    parameters_check = input('## Are the parameters defined in "parameters.json"? (yes/no): ')
    #    if parameters_check.lower() == 'yes':
    #        break
    print('Moving data...')
    move_to_datasets(exp)
else:
    raise FileNotFoundError

    

# Preprocess images, and add them to the 'processed' folder
print('Preprocessing data...')
preprocess()


"""

Calculate Scores and Save Values to CSV File

"""


# Add scores to 'quality_scores.csv'
imgs, file_names = load_processed_images()

csv_file_path = os.path.abspath(os.path.join(
    os.getcwd(),
    os.pardir,
    'datasets',
    'quality_scores.csv'
))

quality_df = pd.read_csv(csv_file_path, dtype=object)
exp_exists = quality_df['Experiment'].tolist()

with open(csv_file_path, "a", newline="") as file:
    writer = csv.writer(file)
    print('Calculating scores...')
    #transformed = get_mean().astype('int16')    
    transformed = io.imread('baseline_mean.png')
    for i, img in enumerate(tqdm(imgs)):
        regex=r'\d*\.?\d+'
        matches = re.findall(regex, file_names[i])
        if matches[0] in exp_exists:
            continue

        img = img.astype('int16')
        #pca, projected, shape = get_pca(imgs, p)
        
        diff_img = abs(transformed-img)
        #diff_img[diff_img < 15] = 0
        #mask = binarize(diff_img)
        #morph = opening(mask, disk(3))
        #blob, areas, eccentricities, perimeters = BLOB(morph)
        quality = int(np.sum(diff_img/1000))
        print("Quality:",quality)
        writer.writerow([matches[0],                # Experiment number
                        matches[1],                # Index
                        matches[2],                # Param 1
                        matches[3],                # Param 2
                        matches[4],                # Param 3
                        matches[5],                # Param 4
                        quality                    # sum of areas (quality score)
                        #areas.shape[0] ,           # Number of BLOBs
                        #np.mean(areas),            # mean area
                        #np.std(areas),             # std area
                        #np.mean(eccentricities),   # mean roundness
                        #np.std(eccentricities),    # std roundness
                        #np.mean(perimeters),       # mean perimeter
                        #np.std(perimeters)])       # std perimeter
                        ])

"""

Bayesian Optimization

"""


# With the new dataset, fit some GP-like model, as suggest a new set of parameters
dims = 4

# Parameter Space
param_bounds = [
    np.linspace(start=0, stop=2, num=21), #
    np.linspace(start=10, stop=100, num=46),
    np.linspace(start=0, stop=5, num=21),
    np.linspace(start=50, stop=300, num=101)
]

param_space = np.array( np.meshgrid(param_bounds[0],
                                    param_bounds[1],
                                    param_bounds[2],
                                    param_bounds[3])).T.reshape(-1,dims)
param_space[:,3] = np.floor(param_space[:,3])

# Define Dataframe
df = pd.read_csv(csv_file_path)

# Initialize model
print('Initialize model...')
#kernel = ConstantKernel() + RBF() + WhiteKernel()
kernel = ConstantKernel() + RationalQuadratic() + WhiteKernel() 
model = GaussianProcessRegressor(kernel=kernel, 
                                 n_restarts_optimizer=10,
                                 random_state=42)

# Define init data
x_init = np.array(df[["Param1", "Param2", "Param3", "Param4"]].iloc[::1,:])
x_init[:,3] = np.floor(x_init[:,3])
y_init = np.array(-df["Area sum"].iloc[::1])

# Fit model on init data
print('Fit model...')
model.fit(standardize(x_init, x_init), standardize(y_init, y_init))

# Calculate best point to sample
print('Calculate best point to sample...')
best_x, ix = opt_acquisition(standardize(x_init, x_init),
                             standardize(x_init, param_space),
                             model)
print(f"Best point: {unstandardize(x_init,best_x)}")


# Calculate estimate and deviation (mean and std)
est_mean, est_std = surrogate(model, [best_x]) # .reshape(-1,1)
est_mean = unstandardize(y_init, est_mean)
est_std  = est_std * (y_init.std()+1e-9)
print(f"Estimated mean: {int(-est_mean)} with deviation: {int(est_std)}")

