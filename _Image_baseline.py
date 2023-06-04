import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from skimage import io

from sklearn.decomposition import PCA, IncrementalPCA

import re

images_path = os.path.abspath(os.path.join(
        os.getcwd(),
        os.pardir,
        'datasets',
        'processed'
    ))

control_exp_nums = ('control_1','control_2') 
""" ('01', '02', '03', '04', '05', '06', '07', '08', '09', 
'10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
'20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
'30', '31', '32', '33', '34', '35', '36', '37', '38', '39', 
'40', '41', '42', '43', '44', '45', '46', '47', '48', '49', 
'50', '51', '52', '53', '54', '55',
'1','2','3','4','5','6','7','8') """

def load_train_images():
    imgs = list(io.imread_collection(images_path+'/*.bmp'))
    return imgs


def get_mean():
    file_names = []
    for filename in os.listdir(images_path):
        file_names.append(filename)
    imgs = load_train_images()
    control_imgs = []
    for i, img in enumerate(imgs):
        #regex=r'\d*\.?\d+'
        #matches = re.findall(regex, file_names[i]) 
        #if matches[0] in control_exp_nums:
        #    control_imgs.append(img)
        if control_exp_nums[0] in file_names[i]:
            control_imgs.append(img)
        if control_exp_nums[1] in file_names[i]:
            control_imgs.append(img)
    return np.array(control_imgs).mean(axis=0).astype(np.uint8)

def get_var():
    file_names = []
    for filename in os.listdir(images_path):
        file_names.append(filename)

    imgs = load_train_images()
    control_imgs = []
    for i, img in enumerate(imgs):
        #regex=r'\d*\.?\d+'
        #matches = re.findall(regex, file_names[i]) 
        #if matches[0] in control_exp_nums:
        #    control_imgs.append(img)
        if control_exp_nums[0] in file_names[i]:
            control_imgs.append(img)
        if control_exp_nums[1] in file_names[i]:
            control_imgs.append(img)
    return np.array(control_imgs).var(axis=0).astype(np.uint8)

def get_pca(imgs, n_components = 3):
    img_shape = imgs[0].shape
    X = np.array(imgs)
    X = X.reshape( (X.shape[0], -1) )

    pca = PCA(n_components=n_components) # , svd_solver='randomized'
    pca.fit(X)
    
    components = pca.transform(X)
    projected = pca.inverse_transform(components)
    
    return pca, projected, img_shape


if __name__ == "__main__":

    from skimage.filters import median
    from pystackreg import StackReg

    
    def stretch(v_min_d, v_max_d, img):
        v_min = np.min(img)
        v_max = np.max(img)
        g = ((v_max_d - v_min_d) / (v_max - v_min)) * (img - v_min) + v_min_d
        return np.round(g).astype('uint8')
    
    dataset_path = os.path.abspath(os.path.join(os.getcwd(),os.pardir,'datasets'))
    raw_path = os.path.join(dataset_path,'processed','train') 
    all_imgs = list(io.imread_collection(raw_path+'/*.bmp'))
    ref_img = median(all_imgs[-2][ 170:750 , 350:950], np.ones((5,5))).astype('int16')
    sr = StackReg(StackReg.RIGID_BODY)

    images_path = os.path.abspath(os.path.join(os.getcwd(),'tests','lighting_door_open'))
    imgs = list(io.imread_collection(images_path+'/*.bmp'))

    mean = np.array(all_imgs).mean(axis=0)#[170:750,350:950] #[220:800, 400:1000] # 
    plt.imshow(mean,cmap='gray')
    plt.colorbar()
    plt.axis(False)
    #plt.clim(0,55)
    plt.show()

    """  no_reg = []
    with_reg = []
    for img in all_imgs: # sum of std: 4.224.280
        img = img[ 170:750 , 350:950]
        img = median(img, np.ones((5,5)) )
        img = img[15:-15, 15:-15]
        img = stretch(0,255, img)
        no_reg.append(img)
    for img in all_imgs: # sum of std: 4.105.951
        img = img[ 170:750 , 350:950]
        img = median(img, np.ones((5,5)) )
        mov = img.astype('int16')
        img = sr.register_transform(ref_img, mov).astype('uint8')
        img = img[15:-15, 15:-15]
        img = stretch(0,255, img)
        with_reg.append(img)
    
    fig, axs = plt.subplots(1,4)
    axs[0].imshow(np.array(no_reg).std(axis=0),cmap='jet')
    axs[0].set_title("No reg")
    axs[1].imshow(np.array(with_reg).std(axis=0),cmap='jet')
    axs[1].set_title("With reg")
    axs[2].imshow(np.array(imgs).std(axis=0),cmap='jet')
    axs[2].set_title("Raw")
    axs[3].imshow(ref_img,cmap='gray')
    axs[3].set_title("Ref")
    print(np.array(no_reg).std(axis=0).sum())
    print(np.array(with_reg).std(axis=0).sum())
    plt.show() """
    


    """  p = 10
    imgs = load_train_images()
    pca, projected, shape = get_pca(imgs, p)

    # Projected data at example #*index*, given only <p> principal components
    plt.imshow(projected[30].reshape(shape),cmap='gray')
    plt.show()
    
    # The n'th principal component(s)
    # fig, axs = plt.subplots(1,5)
    # axs[0].imshow(pca.components_[0].reshape(shape),cmap='gray')
    # axs[1].imshow(pca.components_[1].reshape(shape),cmap='gray')
    # axs[2].imshow(pca.components_[2].reshape(shape),cmap='gray')
    # axs[3].imshow(pca.components_[3].reshape(shape),cmap='gray')
    # axs[4].imshow(pca.components_[4].reshape(shape),cmap='gray')
    # plt.show()
    
    # Explained variance
    plt.plot(np.arange(1,p+1),np.cumsum(pca.explained_variance_ratio_))
    plt.show() """