'''
Wrapper for DewarpNet:
https://github.com/cvlab-stonybrook/DewarpNet
'''

from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
from torch.autograd import Variable
from torch.utils import data

from .models import get_model
from .loaders import get_loader
from .utils import convert_state_dict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------
# Utility functions
# ---------------------------

def error_and_exit(message):
    message = '[bold]ERROR: [/]' + message
    console.print(message, style='red')
    sys.exit(1)

def convert_bgr_to_rgb(image):
    '''Convert input from BGR to RGB'''
    is_grayscale = len(image.shape) == 2
    assert not is_grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------
# Main functions
# ---------------------------

def unwarp(img, bm):
    w,h=img.shape[0],img.shape[1]
    bm = bm.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()[0,:,:,:]
    bm0=cv2.blur(bm[:,:,0],(3,3))
    bm1=cv2.blur(bm[:,:,1],(3,3))
    bm0=cv2.resize(bm0,(h,w))
    bm1=cv2.resize(bm1,(h,w))
    bm=np.stack([bm0,bm1],axis=-1)
    bm=np.expand_dims(bm,0)
    bm=torch.from_numpy(bm).double()

    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).double()

    res = F.grid_sample(input=img, grid=bm, align_corners=False)
    res = res[0].numpy().transpose((1, 2, 0))

    return res


def dewarpnet(img_original, wc_model_path, bm_model_path, verbose=False, debug=False):
    # img_original  : image loaded with cv2.imread; in COLOR_BGR format
    # wc_model_path : filename for world coord regression model
    # bm_model_path : filename for backward mapping regression

    wc_model_path = Path(wc_model_path)
    bm_model_path = Path(bm_model_path)
    
    if not wc_model_path.is_file():
        msg = f'DewarpNet: WC model file not found: {wc_model_path}'
        error_and_exit(msg)
    
    if not bm_model_path.is_file():
        msg = f'DewarpNet: BM model file not found: {bm_model_path}'
        error_and_exit(msg)

    wc_model_name = wc_model_path.stem.split('_')[0]
    bm_model_name = bm_model_path.stem.split('_')[0]

    wc_n_classes = 3
    bm_n_classes = 2

    wc_img_size = (256, 256)
    bm_img_size = (128, 128)

    img_original = convert_bgr_to_rgb(img_original)
    img = cv2.resize(img_original, wc_img_size)
    
    img = img[:, :, ::-1]
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Predict
    htan = nn.Hardtanh(0,1.0)
    wc_model = get_model(wc_model_name, wc_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        wc_state = convert_state_dict(torch.load(wc_model_path, map_location='cpu')['model_state'])
    else:
        wc_state = convert_state_dict(torch.load(wc_model_path)['model_state'])
    wc_model.load_state_dict(wc_state)
    wc_model.eval()
    bm_model = get_model(bm_model_name, bm_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        bm_state = convert_state_dict(torch.load(bm_model_path, map_location='cpu')['model_state'])
    else:
        bm_state = convert_state_dict(torch.load(bm_model_path)['model_state'])
    bm_model.load_state_dict(bm_state)
    bm_model.eval()

    if torch.cuda.is_available():
        wc_model.cuda()
        bm_model.cuda()
        images = Variable(img.cuda())
    else:
        images = Variable(img)

    with torch.no_grad():
        wc_outputs = wc_model(images)
        pred_wc = htan(wc_outputs)
        bm_input = F.interpolate(pred_wc, bm_img_size)
        outputs_bm = bm_model(bm_input)

    # call unwarp
    uwpred = unwarp(img_original, outputs_bm)

    uwpred = uwpred[:,:,::-1]*255

    return uwpred, True  # Image, ok_status

    # Save the output
    #outp=os.path.join(args.out_path,fname)
    #cv2.imwrite(outp,uwpred[:,:,::-1]*255)
