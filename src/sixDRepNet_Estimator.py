import sys, os, argparse
from face_detection import RetinaFace
import torch 
import cv2
from .model import SixDRepNet
import numpy as np
import torchvision
from .FaceBoxes import FaceBoxes
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms
# from FaceBoxes import FaceBoxes
import torch.nn.functional as F
from PIL import Image
from .utils import compute_euler_angles_from_rotation_matrices
from .server import dlogger

class PoseEstimator:
    def __init__(self,args=None,app_name=None):
        self.dlog = dlogger(app_name)
        cudnn.enabled = True
        self.model = SixDRepNet(backbone_name='RepVGG-B1g2',
                   backbone_file='',
                   deploy=True,
                   pretrained=False)
        snapshot_path = 'model/6DRepNet_300W_LP_BIWI.pth'
        saved_state_dict = torch.load(os.path.join(snapshot_path), map_location='cpu')
        if 'model_state_dict' in saved_state_dict:
            self.model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(saved_state_dict)

        self.dlog.debug('Pose Model Load Success')
        self.gpu = torch.has_cuda
        if self.gpu:
            self.model.cuda()
            gpu_id = 0
        else:
            self.model.cpu()
            gpu_id = -1
        self.dlog.debug('Use GPU: {}'.format(self.gpu))
        self.model.eval()
        self.cnn_face_detector = FaceBoxes()
        # self.cnn_face_detector = RetinaFace(gpu_id=gpu_id)
        self.dlog.debug('Face Detection Model Load Success')
        self.transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    def _preprocess(self,img):
        scales = [800,1600]
        im_shape = img.shape
        target_size = scales[0]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        if im_size_max > scales[0]:
            tem = cv2.copyMakeBorder(img, 0, im_size_max-img.shape[0], 0,im_size_max-img.shape[1],cv2.BORDER_CONSTANT, value=(0,0,0))
            im_scale = float(target_size) / float(im_size_max)
        else:
            tem = cv2.copyMakeBorder(img, 0, scales[0]-img.shape[0], 0, scales[0]-img.shape[1],cv2.BORDER_CONSTANT, value=(0,0,0))
            im_scale = 1

        scales = [im_scale]

        im_scale = scales[0]
        if im_scale != 1.0:
            im = cv2.resize(tem,None,None,fx=im_scale,fy=im_scale,interpolation=cv2.INTER_LINEAR)
        else:
            im = tem.copy()
        return im

    def predict(self,img):
        result = []
        # img_scale = self._preprocess(img)
        img_scale = img
        img_rgb = cv2.cvtColor(img_scale,cv2.COLOR_BGR2RGB)
        dets = self.cnn_face_detector(img_scale)
        print(dets)
        # for idx, (det,landmark,score) in enumerate(dets):
        for idx, det in enumerate(dets):
            temp_res = {}
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det[0]
            y_min = det[1]
            x_max = det[2]
            y_max = det[3]
            # conf = det.confidence
            # temp_res['bbox'] = det.tolist()
            temp_res["location"] ={
                "left" : int(x_min),
                "top" : int(y_min),
                "width" : int(abs(x_max-x_min)),
                "height" : int(abs(y_max-y_min))
            }
            # if conf > 1.0:
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = max(x_min, 0); y_min = max(y_min, 0)
            x_max = min(img.shape[1], x_max); y_max = min(img.shape[0], y_max)
            x_min = int(x_min)
            x_max = int(x_max)
            y_min = int(y_min)
            y_max = int(y_max)
            # Crop image
            img_crop = img_rgb[y_min:y_max,x_min:x_max]
            img_crop = Image.fromarray(img_crop)

            # Transform
            img_tensor = self.transformations(img_crop)
            if self.gpu:
                img_tensor = img_tensor.cuda()
            else:
                img_tensor = img_tensor.cpu()
            img_tensor = img_tensor.unsqueeze(0)

            R_pred = self.model(img_tensor)

            euler = compute_euler_angles_from_rotation_matrices(
                R_pred,self.gpu)*180/np.pi
            p_pred_deg = euler[:, 0].cpu().item()
            y_pred_deg = euler[:, 1].cpu().item()
            r_pred_deg = euler[:, 2].cpu().item()
            temp_res["pose"] = {
                "yaw" : y_pred_deg,
                "pitch" : p_pred_deg,
                "roll" : r_pred_deg
            }
            result.append(temp_res)
        return result
if __name__ == '__main__':
    pose_est = PoseEstimator()
    img = cv2.imread(sys.argv[1])
    print(pose_est.predict(img))