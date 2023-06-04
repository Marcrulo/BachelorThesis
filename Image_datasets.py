
import os
import glob
import shutil
import random 
import string
import json 

def get_params(path):
    with open(path,'r') as file:
        return json.load(file)

def move_to_datasets(exp):
    exp_path = os.path.abspath(os.path.join(
        os.getcwd(),
        os.pardir,
        'experiments2',
        exp,
        'images',
    ))

    param_path = os.path.abspath(os.path.join(
        exp_path,
        os.pardir,
        'parameters.json'
    ))
    
    raw_path = os.path.abspath(os.path.join(
        os.getcwd(),
        os.pardir,
        'datasets',
        'raw'
    ))


    # if os.path.exists(train_path):
        # shutil.rmtree(train_path)

    # if os.path.exists(test_path):
        # shutil.rmtree(test_path)

    # os.makedirs( train_path  )
    # os.makedirs( test_path   )
    
    param_dict = get_params(param_path)

    # experiment images
    blacklist = ['26','27','28','29','53','54','55',]
    # 'control_3','control_4','control_5','control_6','control_7','control_8'

    dirs = next(os.walk(exp_path))[1]
    dir_names = [d for d in dirs]

    for dir in dir_names:
        if dir in blacklist:
            continue
        params_name = f"{str(param_dict[dir])}"
        
        dir_path = os.path.abspath(os.path.join(exp_path,dir))
        images = glob.glob(dir_path + "/*.bmp")

        for i, src_file in enumerate(images):
            name = f"{dir},{i},{params_name}"
            dst_file = os.path.join(raw_path, name+'.bmp')
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copyfile(src_file, dst_file)

        
        
if __name__ == "__main__":
    exps = ['Exp0_20feb']#['Exp0_25jan']
    for exp in exps:
        move_to_datasets(exp)
