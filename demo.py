from face_detection import RetinaFace
from src.model import SixDRepNet
import math
import re
from matplotlib import pyplot as plt
import sys
import os
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.function_base import _quantile_unchecked

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import src.utils as utils
import matplotlib
from PIL import Image
import time
matplotlib.use('Agg')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cam',
                        dest='cam_id', help='Camera device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)
    parser.add_argument('--image',default='images/temp.jpg',type=str)
    args = parser.parse_args()
    return args


transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    cam = args.cam_id
    if_image = True
    snapshot_path = args.snapshot
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    print('Loading data.')

    detector = RetinaFace(gpu_id=gpu)

    # Load snapshot
    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    if gpu <= 0 :
        model.cpu()
    else:
        model.cuda(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    if if_image == False:
        cap = cv2.VideoCapture(cam)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        with torch.no_grad():
            while True:
                ret, frame = cap.read()

                faces = detector(frame)

                for box, landmarks, score in faces:

                    # Print the location of each face in this image
                    if score < .95:
                        continue
                    x_min = int(box[0])
                    y_min = int(box[1])
                    x_max = int(box[2])
                    y_max = int(box[3])
                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)

                    x_min = max(0, x_min-int(0.2*bbox_height))
                    y_min = max(0, y_min-int(0.2*bbox_width))
                    x_max = x_max+int(0.2*bbox_height)
                    y_max = y_max+int(0.2*bbox_width)

                    img = frame[y_min:y_max, x_min:x_max]
                    img = Image.fromarray(img)
                    img = img.convert('RGB')
                    img = transformations(img)

                    if gpu<=0:
                        img = torch.Tensor(img[None, :]).cpu()
                    else:
                        img = torch.Tensor(img[None, :]).cuda(gpu)

                    c = cv2.waitKey(1)
                    if c == 27:
                        break

                    start = time.time()
                    R_pred = model(img)
                    end = time.time()
                    print('Head pose estimation: %2f ms' % ((end - start)*1000.))

                    euler = utils.compute_euler_angles_from_rotation_matrices(
                        R_pred)*180/np.pi
                    p_pred_deg = euler[:, 0].cpu()
                    y_pred_deg = euler[:, 1].cpu()
                    r_pred_deg = euler[:, 2].cpu()

                    #utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
                    utils.plot_pose_cube(frame,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
                        x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)

                cv2.imshow("Demo", frame)
                cv2.waitKey(5)
    else:
        image = cv2.imread(args.image)    
        with torch.no_grad():
            faces = detector(image)

            for box, landmarks, score in faces:

                # Print the location of each face in this image
                if score < .95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = image[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)

                if gpu<=0:
                    img = torch.Tensor(img[None, :]).cpu()
                else:
                    img = torch.Tensor(img[None, :]).cuda(gpu)

                start = time.time()
                R_pred = model(img)
                end = time.time()
                print('Head pose estimation: %2f ms' % ((end - start)*1000.))

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred,use_gpu=False)*180/np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                # utils.draw_axis(image, y_pred_deg, p_pred_deg, r_pred_deg, x_min+int(.5*(x_max-x_min)), y_min, size=100)
                utils.plot_pose_cube(image,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
                    x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)

            # cv2.imwrite('temp_out.py',image)
            cv2.imshow("Demo", image)
            cv2.waitKey()