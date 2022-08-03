from pathlib import Path

from .seg import U2NETP
from .GeoTr import GeoTr
from .IllTr import IllTr
from .inference_ill import rec_ill

import torch
import torch.nn as nn
import torch.nn.functional as F
#import skimage.io as io
import numpy as np
import cv2
#import glob
#import os
#from PIL import Image
#import argparse

import warnings
warnings.filterwarnings('ignore')

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)
        
    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x

        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        
        return bm
        

def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        #print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        #print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
        

def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        #print(len(pretrained_dict.keys()))
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        #print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
        

def initialize(model_path, model_dict, verbose=False):
    
    if 'DocTr' in model_dict:
        return model_dict['DocTr']

    if verbose:
        print('Loading pre-trained DocTr model', flush=True)

    model_path = Path(model_path)
    Seg_path = model_path / 'seg.pth'
    GeoTr_path = model_path / 'geotr.pth'
    IllTr_path = model_path / 'illtr.pth'

    GeoTr_Seg_model = GeoTr_Seg().cuda() if torch.cuda.is_available() else GeoTr_Seg().cpu()
    # reload segmentation model
    reload_segmodel(GeoTr_Seg_model.msk, Seg_path)
    # reload geometric unwarping model
    reload_model(GeoTr_Seg_model.GeoTr, GeoTr_path)
    
    IllTr_model = IllTr().cuda() if torch.cuda.is_available() else IllTr().cpu()
    # reload illumination rectification model
    reload_model(IllTr_model, IllTr_path)
    
    # To eval mode
    GeoTr_Seg_model.eval()
    IllTr_model.eval()

    if verbose:
        print('   ... model loaded', flush=True)

    model_dict['DocTr'] = GeoTr_Seg_model, IllTr_model
    return model_dict['DocTr']


def DocTr(img_original, model_path, model_dict, rectify_illumination=False, verbose=False, debug=False):
    GeoTr_Seg_model, IllTr_model = initialize(model_path, model_dict, verbose)
    im_ori = img_original[:, :, :3] / 255 # BUGBUG??
    h, w, _ = im_ori.shape
    im = cv2.resize(im_ori, (288, 288))
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float().unsqueeze(0)
    
    with torch.no_grad():
        # geometric unwarping
        bm = GeoTr_Seg_model(im.cuda() if torch.cuda.is_available() else im.cpu())
        bm = bm.cpu()
        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2
        
        out = F.grid_sample(torch.from_numpy(im_ori).permute(2,0,1).unsqueeze(0).float(), lbl, align_corners=True)
        img_geo = ((out[0]*255).permute(1, 2, 0).numpy())[:,:,::-1].astype(np.uint8)

        # AT THIS POINT WE HAVE RECTIFIED THE GEOMETRY AND COULD STOP
        
        # illumination rectification
        if rectify_illumination:
            img_geo = rec_ill(IllTr_model, img_geo)

        return img_geo, True


if __name__ == '__main__':
    pass # TODO
