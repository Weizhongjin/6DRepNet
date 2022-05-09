from __future__ import print_function

import torch
import os
from .data import cfg_mnet,cfg_re50
from .models.retinaface import RetinaFace


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(pretrained_path):
    # current_path = os.path.abspath(__file__)
    # father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    # pretrained_path = father_path + '/weights/mobilenet0.25_Final.pth'
    # print('Loading pretrained model from {}'.format(pretrained_path))

    model = RetinaFace(cfg=cfg_re50, phase='test')
    #device = torch.cuda.current_device()
    #pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    # print('Finished loading model!')
    return model
