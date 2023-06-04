import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from skimage import io, color, exposure
from skimage.filters import prewitt, median, gaussian
from skimage.morphology import disk, dilation, erosion, closing, opening 
from skimage.transform import rescale

from pystackreg import StackReg

from PIL import Image

import random
import string
import shutil
from tqdm import tqdm

from copy import deepcopy

def preprocess():

    dataset_path = os.path.abspath(os.path.join(
        os.getcwd(),
        os.pardir,
        'datasets'
    ))

    # input images path
    raw_path = os.path.join(dataset_path,'raw') 
    
    # save processed images
    processed_path = os.path.join(dataset_path,'processed')
    
    # if os.path.exists(processed_path):
        # shutil.rmtree(processed_path)
    # os.makedirs( processed_path )
    
    # list of images
    imgs = list(io.imread_collection(raw_path+'/*.bmp'))
    file_names = []
    for filename in os.listdir(raw_path):
        file_names.append(filename)
    file_names.sort()
        

    # Only check raw images that are not in processed
    raw_files =  os.listdir(raw_path)
    proc_files = os.listdir(processed_path)
    diffs = [element for element in raw_files if element not in proc_files]
    indices = [i for i, element in enumerate(file_names) if element in diffs]
    imgs = [imgs[i] for i in indices]
    file_names = [file_names[i] for i in indices]

    def stretch(v_min_d, v_max_d, img):
        v_min = np.min(img)
        v_max = np.max(img)
        g = ((v_max_d - v_min_d) / (v_max - v_min)) * (img - v_min) + v_min_d
        return np.round(g).astype('uint8')
    

    #ref_img = median(all_imgs[-2][ 170:750 , 350:950], np.ones((5,5))).astype('int16')
    ref_img = io.imread('baseline_ref.png').astype('int16')
    mean_img = io.imread('baseline_mean.png')
    sr = StackReg(StackReg.RIGID_BODY)
    
    for i, img in enumerate(tqdm(imgs)):
        # CROPPING
        img = img[ 220:800 , 350:950]
        #img = img[ 170:750 , 350:950]
        
        # MEDIAN FILTER 
        img = median(img, np.ones((5,5)) )
        
        # RIDIG REGISTRATION
        mov = img.astype('int16')
        img = sr.register_transform(ref_img, mov).astype('uint8')
        
        # Crop little errors on the sides
        img = img[15:-15, 15:-15]

        # HISTOGRAM STRETCHING
        img = stretch(0,255, img)
        
        # HISTOGRAM MATCHING
        img = exposure.match_histograms(img, mean_img)

        save_img = Image.fromarray(img).convert('L')
        name = file_names[i]
        dst_file = os.path.join(processed_path, name)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        save_img.save( dst_file )    

if __name__ == "__main__":
    preprocess()