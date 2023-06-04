import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from skimage import io, measure

from skimage.morphology import disk, dilation, erosion, closing, opening, remove_small_objects
from skimage import exposure
import csv
import json
import re
from tqdm import tqdm
from PIL import Image, ImageOps

def load_processed_images():
    dataset_path = os.path.abspath(os.path.join(
        os.getcwd(),
        os.pardir,
        'datasets',
        'processed',
    ))
    
    imgs = list(io.imread_collection(dataset_path+'/*.bmp'))
    file_names = []
    for filename in os.listdir(dataset_path):
        file_names.append(filename)

    file_names.sort()

    return imgs, file_names

def BLOB(bin_img):
    label_img = measure.label(bin_img, connectivity=2)
    #n_labels = label_img.max()
    
    region_props = measure.regionprops(label_img)
    
    areas = []
    eccentricities = []
    perimeters = []
    for region in region_props:
        areas.append(region.area)
        eccentricities.append(region.eccentricity)
        perimeters.append(region.perimeter)
    
    return label_img, np.array(areas), np.array(eccentricities), np.array(perimeters)

def stretch(v_min_d, v_max_d, img):
    v_min = np.min(img)
    v_max = np.max(img)
    g = ((v_max_d - v_min_d) / (v_max - v_min)) * (img - v_min) + v_min_d
    return np.round(g).astype('int16')
    
def binarize(img, T=20):
    mask = img > T#( (img > T )*255).astype('uint8')
    return mask


if __name__ == "__main__":
    imgs, file_names = load_processed_images()
    csv_file_path = os.path.abspath(os.path.join(
        os.getcwd(),
        os.pardir,
        'datasets',
        'quality_scores.csv'
    ))
    
    testing = True

    if testing:
        for i, img in enumerate(imgs):
            regex=r'\d*\.?\d+'
            matches = re.findall(regex, file_names[i])
            inspect = ('23','7') # exp, index

            
            if matches[0] == inspect[0] and matches[1] == inspect[1]:
                img = img.astype('int32')
                
                
                # p = 100
                # pca, projected, shape = get_pca(imgs, p)
                # img = projected[i].reshape(shape)
                mean_img = io.imread('baseline_mean.png')
                transformed = stretch(0,255,mean_img.astype('int32'))
                #transformed = projected[i].reshape(shape)
                matched = exposure.match_histograms(img, transformed, multichannel=False)
                diff_img = transformed-matched
                cutoff_diff_img = diff_img.copy()
                cutoff_diff_img[abs(diff_img)<20] = 0
                mask = binarize(diff_img)
                morph = opening(mask, disk(3))
                blob, areas, eccentricities, perimeters = BLOB(morph)
                

                fig, axs = plt.subplots(2,3,figsize=(5,5))
                axs[0,0].imshow(matched,cmap='gray')
                axs[0,0].set_title(f"Original {inspect[0]}-{inspect[1]}")

                axs[0,1].imshow(transformed,cmap='gray')
                axs[0,1].set_title("Baseline")

                #axs[0,2].imshow(img*0,cmap='gray')
                #axs[0,2].set_title("")
                # axs[1,0].hist(img.ravel(),bins=50)
                # axs[1,0].set_ylim(0,45000)
                # axs[1,0].set_title("Hist")

                #score = axs[0,2].imshow(transformed-img, cmap='RdYlBu')
                score = axs[0,2].imshow(cutoff_diff_img, cmap='RdYlBu')
                axs[0,2].set_title(f"Score: {int(np.sum(abs(cutoff_diff_img))/1000)}")
                fig.colorbar(score, ax=axs[0,2])
                score.set_clim(vmin=-80, vmax=80)
                #axs[1,1].hist(transformed.ravel(),bins=50)
                #axs[1,1].set_ylim(0,45000)
                #axs[1,1].set_title(f"Score: {int(np.sum(diff_img)/1000)}") 

                upper_bound = 85000
                #axs[1,0].imshow(matched, cmap='gray')

                axs[1,0].hist(matched.ravel(),bins=20)
                axs[1,0].set_ylim(0,upper_bound)
                axs[1,1].hist(transformed.ravel(),bins=20)
                axs[1,1].set_ylim(0,upper_bound)
                axs[1,2].hist((transformed-img).ravel(),bins=50)

                # morph = opening(mask, disk(2))
                # blob, areas, eccentricities, perimeters = BLOB(morph)
                # axs[1,0].imshow(morph,cmap='jet')
                # axs[1,0].set_title(f"Score: {np.sum(areas)}")

                # morph = opening(mask, disk(3))
                # blob, areas, eccentricities, perimeters = BLOB(morph)
                # axs[1,1].imshow(morph,cmap='jet')
                # axs[1,1].set_title(f"Score: {np.sum(areas)}")

                # morph = opening(mask, disk(4))
                # blob, areas, eccentricities, perimeters = BLOB(morph)
                # axs[1,2].imshow(morph,cmap='jet')
                # axs[1,2].set_title(f"Score: {np.sum(areas)}")

                plt.show()
                #break

    else:   
        with open(csv_file_path, "w+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Experiment",
                            "Index",
                            "Param1",
                            "Param2",
                            "Param3",
                            "Param4",
                            "Param5",
                            "Param6", 
                            "BLOBs",
                            "Area sum",
                            "Area mean" ,
                            "Area std",
                            "Roundness mean", 
                            "Roundness std",
                            "Perimeter mean",
                            "Perimeter std"])
            
            for i, img in enumerate(imgs):
                regex=r'\d*\.?\d+'
                matches = re.findall(regex, file_names[i])
                
                img = img.astype('int16')
                #pca, projected, shape = get_pca(imgs, p)
                mean_img = io.imread('baseline_mean.png')
                transformed = mean_img.astype('int16')
                diff_img = abs(transformed-img)
                mask = binarize(diff_img)
                morph = opening(mask, disk(3))
                blob, areas, eccentricities, perimeters = BLOB(morph)
                
                writer.writerow([matches[0],                # Experiment number
                                matches[1],                # Index
                                matches[2],                # Param 1
                                matches[3],                # Param 2
                                matches[4],                # Param 3
                                matches[5],                # Param 4
                                matches[6],                # Param 5
                                matches[7],                # Param 6
                                areas.shape[0] ,           # Number of BLOBs
                                int(np.sum(diff_img/1000)),# sum of areas (quality score)
                                np.mean(areas),            # mean area
                                np.std(areas),             # std area
                                np.mean(eccentricities),   # mean roundness
                                np.std(eccentricities),    # std roundness
                                np.mean(perimeters),       # mean perimeter
                                np.std(perimeters)])       # std perimeter
