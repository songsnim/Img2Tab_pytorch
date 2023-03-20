import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models.psp import pSp
from xgboost import XGBClassifier
from models.stylegan2.model import StyleSpace
from utils.demo_utils import *
from argparse import Namespace

def load_GAN_inversion(device):
    model_path = "pretrained_models/e4e_ffhq_encode.pt"
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts['device'] = device
    opts= Namespace(**opts)
    opts.channel_multiplier=2

    net = pSp(opts)
    net.eval()
    net.to(device)
    return net

def load_classifier(attr):
    model_dir = f'pretrained_models/xgboost/Final_{attr.lower()}_XGBoost.json'
    classifier = XGBClassifier(tree_method='gpu_hist')
    classifier.load_model(model_dir)
    return classifier

def feed_forward_once(net, image_tensor, global_mode, device, return_latent=False):
    if global_mode == "get_stylespace":
        StyleSpace.stage = 0
        StyleSpace.stylespaces = {}
        StyleSpace.test_stage = 0
        StyleSpace.mode = global_mode
    elif global_mode == "test":
        StyleSpace.mode = "test"
        StyleSpace.test_stage = 0

    image = image_tensor
    output_image, output_latent = net(image.unsqueeze(0),randomize_noise=False,return_latents=True)
    sample_stylespace = torch.cat(list(StyleSpace.stylespaces.values()), dim=1) # flatten for model input
    
    if return_latent:
        return sample_stylespace, output_image, output_latent
    else:
        return sample_stylespace, output_image

class Img2Tab(nn.Module):
    def __init__(self, attr, device):
        super().__init__()
        self.attr = attr 
        self.device = device
        self._inversion_net = load_GAN_inversion(self.device)
        self.classifiers = load_classifier(self.attr)
        self.StyleSpace = StyleSpace
        print('Img2Tab networks have been loaded successfully!')
    
    def __str__(self):
        return "Img2Tab framework to predict an image and explain at the concept level."
    
    def get_psi_vector(self, images):
        psi_vector, x = feed_forward_once(self._inversion_net, images, 'get_stylespace', self.device)
        return psi_vector
    
    def predict(self, image, _class):
        psi_vector = self.get_psi_vector(image) # psi_i = E(x_i)
        prediction = self.classifiers.predict(to_np(psi_vector)) # y_i=P(psi_i)
        if prediction.item() == 1:
            print(f"This person is perceived as {_class}.")
        else:
            print(f"This person is perceived as not {_class}.")
        return prediction
    
    def get_important_concepts(self, n):
        imps = list(self.classifiers.feature_importances_)
        important_concepts_indices = [imps.index(x) for x in sorted(imps, reverse=True)][:n]
        
        return imps, important_concepts_indices
    
    def _denorm_for_visualizing(self, image):
        return denorm(to_np(image.squeeze(0).permute(1,2,0)))
    
    def visualize_concepts(self, image, k_concepts, lambda_list):
        dim_list = self.StyleSpace.stylespace_dim
        with torch.no_grad():
            fig = plt.figure(figsize=(3*len(k_concepts),3), dpi=300)
            for k_idx, (k, lambda_) in enumerate(zip(k_concepts, lambda_list)):
                l_idx, s_idx = get_l_s_index(k, dim_list)
                if StyleSpace.stylespaces == {}:
                    feed_forward_once(self._inversion_net, image, 'get_stylespace', self.device)
                StyleSpace.stylespaces[l_idx][:,s_idx] += lambda_ #latent modification
                _, output_image = feed_forward_once(self._inversion_net, image, 'test', self.device)
                _, source = feed_forward_once(self._inversion_net, image, 'get_stylespace', self.device) #reset

                if k_idx == 0:
                    plt.subplot(1,len(k_concepts)+1,1)
                    plt.title('source')
                    plt.axis('off')
                    plt.imshow(self._denorm_for_visualizing(source))
                
                plt.subplot(1,len(k_concepts)+1,k_idx+2)
                plt.title(f'{k}th feature')
                plt.axis('off')
                plt.imshow(self._denorm_for_visualizing(output_image))
    

    def _train_classifier(self, x_train, y_train):
        params = {
            'learning_rate':0.1,
            'eta':0.1,
            'n_estimators':600,
            'max_depth':20,
            'min_child_weight':1,
            'gamma':0.3,
            'subsample': 0.74,
            'colsample_bytree':0.6,
            'reg_alpha':10,
            'reg_lambda': 0.1, # or 10 
            'random_state': 2000
        }
        gender_model = XGBClassifier(**params, tree_method='gpu_hist', gpu_id=1)
        xgb_model = gender_model.fit(x_train, y_train, verbose=False)
        
        self.classifier = xgb_model
        
    
    def _mask_unwanted_concepts(self, Psi_matrix, unwanted_k):
        for k in unwanted_k:
            Psi_matrix[:, k] = 0
            
        return Psi_matrix
    
    def debug_classifier(self, unwanted_k, Psi_matrix, label):
        masked_Psi_matrix = self._mask_unwanted_concepts(Psi_matrix, unwanted_k)
        self._train_classifier(masked_Psi_matrix, label)
        
        
        
        