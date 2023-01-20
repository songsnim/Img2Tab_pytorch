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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resize_dims = (256, 256)
img_transforms = transforms.Compose([
    transforms.Resize(resize_dims),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def run_alignment(image_path):
    from utils.alignment import align_face
    predictor = dlib.shape_predictor(
        "pretrained_models/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    # print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def display_alongside_source_image(result_image, source_image):
    res = np.concatenate([np.array(result_image.resize(resize_dims)),
                          np.array(source_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)


def run_on_batch(inputs, net):
    images, latents = net(inputs.to(device).float(),
                          randomize_noise=False, return_latents=True)

    return images, latents

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
        
def image_path_to_tensor(image_path, data_type, align=True):
    if data_type == 'face':
        img_transforms = transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        input_image = run_alignment(image_path) if align else Image.open(image_path)
        input_image.resize((256,256))
        image_tensor = img_transforms(input_image)
        
    elif data_type == 'mnist':
        img_transforms = transforms.Compose(
            [transforms.Resize((32, 32)),
            transforms.ToTensor(),
			transforms.Normalize([0.5], [0.5])]
        )
        
        input_image = Image.open(image_path)
        image_tensor = img_transforms(input_image)
        image_tensor = image_tensor.repeat(3,1,1)
    
    return image_tensor

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

def infer_model(net, model, image_tensor, align):
    image = image_tensor.to(device)
    sample_stylespace, output_image = feed_forward_once(net, image, 'get_stylespace', device)
    sample_stylespace = to_np(sample_stylespace)
    # sample_stylespace = to_np(torch.cat(list(Global.stylespaces.values()), dim=1))
    if str(model)[:6] == 'TabNet':
        TabNet = model
        pred = TabNet.predict(sample_stylespace)
        proba = TabNet.predict_proba(sample_stylespace)
        explain_matrix, masks = TabNet.explain(sample_stylespace)
        imps = list(explain_matrix[0])
        cores = [imps.index(x) for x in sorted(imps, reverse=True)][:5]
        # 1이면 attr, 0이면 not attr
        return pred, cores, proba, explain_matrix
        
    else:
        xgb = model
        pred = xgb.predict(sample_stylespace)
        imps = list(xgb.feature_importances_)
        cores = [imps.index(x) for x in sorted(imps, reverse=True)][:5]
        proba = xgb.predict_proba(sample_stylespace)
    
        return pred, cores, proba, None

def normalize(style):
    mean = style.mean(axis=0)
    std = style.std(axis=0)
    return (style - mean) / std

def get_normalized_stylespaces(x_train):
    normalized = [] 
    for idx in tqdm(range(9088)):
        style = x_train[:, idx]
        normalized.append(normalize(style))
    stacked = np.stack(normalized, axis=1)
    return stacked 
    
def get_wasser_list(stylespaces, label):
    pos_idx = np.where(label==1)[0]
    neg_idx = np.where(label==0)[0]
    normed_styles = get_normalized_stylespaces(stylespaces)
    impurity_list = []
    for core in tqdm(range(stylespaces.shape[1])):
        p = normed_styles[pos_idx,core]
        q = normed_styles[neg_idx,core]
        wassers = wasserstein_distance(p, q)
        impurity_list.append(wassers)
    return impurity_list


def wandb_save_images(gender, degree, mode, description):
    import wandb
    image_list = os.listdir(f"ffhq-dataset/CelebA_gender_test/{gender}/")
    wandb.init(project='Img2Tab', entity='songyj', name=description)
    wandb.log({"image": wandb.Image(image)})
    pass

if __name__ == '__main__':
    pass