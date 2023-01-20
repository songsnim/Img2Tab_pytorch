import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torchvision
import torchvision.transforms as transforms

from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

import PIL
from PIL import Image
import dlib

from scipy.stats import wasserstein_distance
from tqdm import tqdm

from models.stylegan2.model import StyleSpace

def run_alignment(image_path):
    from utils.alignment import align_face
    predictor = dlib.shape_predictor(
        "pretrained_models/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    # print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

def to_np(tensor):
    return tensor.cpu().detach().numpy()

def denorm(tensor, mode='clip'):
    if mode == 'clip':
        return np.clip(np.array((tensor+1)/2), 0, 1)
    
    elif mode == 'minmax':
        _max = tensor.max()
        _min = tensor.min()
        return (tensor-_min)/(_max-_min)
    
    elif mode == 'standard':
        
        mean = tensor.mean()
        std = tensor.std()
        return (tensor-mean)/std

def get_l_s_index(core, dim_list=None):
    if dim_list == None:
        dim_list = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 
                    512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32]
    for idx, dim in enumerate(dim_list):
        if core > dim:
            core -= dim
        else: 
            return idx, core

def get_image_from_path(path, align=True, show=True):
    img_transforms = transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
    input_image = run_alignment(path) if align else Image.open(path)
    input_image.resize((256,256))
    image_tensor = img_transforms(input_image)
    
    if show:
        plt.figure(figsize=(2,2))
        plt.axis('off')
        plt.imshow(denorm(image_tensor.permute(1,2,0)))
    return image_tensor

def standardize(style):
    mean = style.mean(axis=0)
    std = style.std(axis=0)
    return (style - mean) / std

def standardize_column_vectors(Psi_matrix):
    standardized = [] 
    for idx in tqdm(range(Psi_matrix.shape[1])):
        column_vector = Psi_matrix[:, idx]
        standardized.append(standardize(column_vector))
    stacked = np.stack(standardized, axis=1)
    print(f'Column vector standardization complete!')
    return stacked

def get_Psi_label(attr):
    Psi_label = to_np(torch.load(f'datasets/Psi_and_labels/{attr}_train_labels.pt'))
    return Psi_label

def get_Wk_list(label, Psi_matrix, Psi_is_standardized):
    if Psi_is_standardized:
        pass
    else:
        Psi_matrix = standardize_column_vectors(Psi_matrix)
        
    pos_idx = np.where(label==1)[0]
    neg_idx = np.where(label==0)[0]
    Wk_list = []
    
    for feature_k in tqdm(range(Psi_matrix.shape[1])):
        Psi_pos = Psi_matrix[pos_idx, feature_k]
        Psi_neg = Psi_matrix[neg_idx, feature_k]
        Wk = wasserstein_distance(Psi_pos, Psi_neg)
        Wk_list.append(Wk)
    
    return Wk_list

if __name__ == '__main__':
    pass