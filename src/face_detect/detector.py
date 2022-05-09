from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .data import cfg_mnet
from .layers.functions.prior_box import PriorBox
from .loader import load_model
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms
import time
import sklearn
import cv2
from sklearn import preprocessing
from skimage import transform as trans


class face_det_model(object):
    def __init__(self, model_path):
        # cudnn.benchmark = True
        # 对于不同输入尺寸，一定要设置成false
        cudnn.benchmark = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        self.model = load_model(model_path).to(self.device)
        self.model.eval()
        print('finish load face detctor model...')

        # params set
        # 默认不向外填充
        self.crop_ratio = 0.0
        self.crop_margin = 0

    def detect_result_face(self, ori_img):
        """[summary]

        Args:
            ori_img ([type]): [输入原始图像，对原始图像进行人脸检测，可设置是否resize]

        Returns:
            [list]: [返回矫正后的人脸,扩充后的人脸检测框[xmin,ymin,xmax,ymax,confidence]]
        """
        image = ori_img.copy()

        # cal resize
        target_size = 640
        max_size = 1024
        im_shape = image.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if resize > 1:
            resize = 1

        bounding_boxes, points = self.detect_faces(image, resize=resize)

        face_list = []
        for i in range(bounding_boxes.shape[0]):
            width = int(bounding_boxes[i][2]) - int(bounding_boxes[i][0])
            height = int(bounding_boxes[i][3]) - int(bounding_boxes[i][1])
            # print height,width
            bx1 = np.maximum(int(bounding_boxes[i][0] - width * self.crop_ratio), 0)
            by1 = np.maximum(int(bounding_boxes[i][1] - height * self.crop_ratio), 0)
            bx2 = np.minimum(int(bounding_boxes[i][2] + width * self.crop_ratio), image.shape[1])
            by2 = np.minimum(int(bounding_boxes[i][3] + height * self.crop_ratio), image.shape[0])
            confidence = bounding_boxes[i][4]

            align_point = points[i].copy()
            # 人脸对齐 3*112*112, RGB
            crop_face = self.align_face(image, align_point)

            # bx1_ = np.maximum(int(bounding_boxes[i][0] - self.crop_margin), 0)
            # by1_ = np.maximum(int(bounding_boxes[i][1] - self.crop_margin), 0)
            # bx2_ = np.minimum(int(bounding_boxes[i][2] + self.crop_margin), image.shape[1])
            # by2_ = np.minimum(int(bounding_boxes[i][3] + self.crop_margin), image.shape[0])
            # save_face = image[by1_:by2_, bx1_:bx2_].copy()

            # 返回人脸图像+位置信息+矫正前原图
            # face_list.append((crop_face, [bx1,by1,bx2,by2], save_face))

            # 返回人脸图像+位置信息
            face_list.append((crop_face, [bx1, by1, bx2, by2, confidence], points))

        return face_list

    def detect_multi_face_2(self, ori_img,bounding_boxes, points):
        """[summary]

        Args:
            ori_img ([type]): [输入原始图像，对原始图像进行人脸检测，可设置是否resize]

        Returns:
            [list]: [返回矫正后的人脸,扩充后的人脸检测框[xmin,ymin,xmax,ymax,confidence]]
        """
        image = ori_img.copy()

        # cal resize
        target_size = 640
        max_size = 1024
        im_shape = image.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if resize > 1:
            resize = 1


        face_list = []
        for i in range(bounding_boxes.shape[0]):
            width = int(bounding_boxes[i][2]) - int(bounding_boxes[i][0])
            height = int(bounding_boxes[i][3]) - int(bounding_boxes[i][1])
            # print height,width
            bx1 = np.maximum(int(bounding_boxes[i][0] - width * self.crop_ratio), 0)
            by1 = np.maximum(int(bounding_boxes[i][1] - height * self.crop_ratio), 0)
            bx2 = np.minimum(int(bounding_boxes[i][2] + width * self.crop_ratio), image.shape[1])
            by2 = np.minimum(int(bounding_boxes[i][3] + height * self.crop_ratio), image.shape[0])
            # confidence = bounding_boxes[i][4]

            align_point = points[i].copy()
            # 人脸对齐 3*112*112, RGB
            crop_face = self.align_face(image, align_point)

            # bx1_ = np.maximum(int(bounding_boxes[i][0] - self.crop_margin), 0)
            # by1_ = np.maximum(int(bounding_boxes[i][1] - self.crop_margin), 0)
            # bx2_ = np.minimum(int(bounding_boxes[i][2] + self.crop_margin), image.shape[1])
            # by2_ = np.minimum(int(bounding_boxes[i][3] + self.crop_margin), image.shape[0])
            # save_face = image[by1_:by2_, bx1_:bx2_].copy()

            # 返回人脸图像+位置信息+矫正前原图
            # face_list.append((crop_face, [bx1,by1,bx2,by2], save_face))

            # 返回人脸图像+位置信息
            face_list.append((crop_face, [bx1, by1, bx2, by2], points))

        return face_list, bounding_boxes, points

    def detect_multi_face(self,ori_img):
        """[summary]

        Args:
            ori_img ([type]): [输入原始图像，对原始图像进行人脸检测，可设置是否resize]

        Returns:
            [list]: [返回矫正后的人脸,扩充后的人脸检测框[xmin,ymin,xmax,ymax,confidence]]
        """
        image = ori_img.copy()
        
        # cal resize
        target_size = 640
        max_size = 1024
        im_shape = image.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if resize > 1:
            resize = 1

        bounding_boxes, points = self.detect_faces(image,resize=resize)

        face_list = []
        for i in range(bounding_boxes.shape[0]):
            width = int(bounding_boxes[i][2]) - int(bounding_boxes[i][0])
            height = int(bounding_boxes[i][3]) - int(bounding_boxes[i][1])
            # print height,width
            bx1 = np.maximum(int(bounding_boxes[i][0] - width  * self.crop_ratio), 0)
            by1 = np.maximum(int(bounding_boxes[i][1] - height * self.crop_ratio), 0)
            bx2 = np.minimum(int(bounding_boxes[i][2] + width  * self.crop_ratio), image.shape[1])
            by2 = np.minimum(int(bounding_boxes[i][3] + height * self.crop_ratio), image.shape[0])
            confidence = bounding_boxes[i][4] 

            align_point = points[i].copy()
            # 人脸对齐 3*112*112, RGB
            crop_face = self.align_face(image,align_point)

            # bx1_ = np.maximum(int(bounding_boxes[i][0] - self.crop_margin), 0)
            # by1_ = np.maximum(int(bounding_boxes[i][1] - self.crop_margin), 0)
            # bx2_ = np.minimum(int(bounding_boxes[i][2] + self.crop_margin), image.shape[1])
            # by2_ = np.minimum(int(bounding_boxes[i][3] + self.crop_margin), image.shape[0])
            # save_face = image[by1_:by2_, bx1_:bx2_].copy()

            # 返回人脸图像+位置信息+矫正前原图
            # face_list.append((crop_face, [bx1,by1,bx2,by2], save_face))

            # 返回人脸图像+位置信息
            face_list.append((crop_face,[bx1,by1,bx2,by2,confidence],points))

        return face_list,bounding_boxes,points


    def detect_face_for_gallery(self,ori_img):
        """ 
            对应初始化gallery的人脸检测接口
            Input:一张图像（认为这种图像中最中间并且比较大的脸为gallery）
            Return：裁剪后符合gallery标准的人脸

        Args:
            image ([type]): [description]

        Returns:
            [type]: [description]
        """
        image = ori_img.copy()
        bounding_boxes, points = self.detect_faces(image)

        num_faces = bounding_boxes.shape[0]
        # 判断检测到的人脸距离图像中心的偏移及人脸大小 —> 决定选取哪张脸；
        crop_face = None
        if num_faces>0:
            bbindex = 0
            if num_faces > 1:
                img_size = np.asarray(image.shape)[0:2]
                det = bounding_boxes[:,0:4]
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                bbindex = np.argmax(bounding_box_size-offset_dist_squared*2.0)
            dst_points = points[bbindex]
            crop_face = self.align_face(image,dst_points)

        return crop_face


    def align_face(self,image,landmark):
        """[summary]
            矫正到标准人脸
        Args:
            image ([type]): [description]
            points ([type]): [description]
        """

        #标准人脸
        self.src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        self.src[:, 0] += 8.0
        self.image_size = [112,112]

        assert landmark.shape[0] == 5 and landmark.shape[1] == 2

        tform = trans.SimilarityTransform()
        tform.estimate(landmark, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(image, M, (self.image_size[1], self.image_size[0]),borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        return img


    def detect_faces(self, img_raw, confidence_threshold=0.9, top_k=5000, nms_threshold=0.4, keep_top_k=750, resize=1):
        """[summary]

        Args:
            img_raw ([type]): [description]
            confidence_threshold (float, optional): [description]. Defaults to 0.9.
            top_k (int, optional): [description]. Defaults to 5000.
            nms_threshold (float, optional): [description]. Defaults to 0.4.
            keep_top_k (int, optional): [description]. Defaults to 750.
            resize (int, optional): [description]. Defaults to 1.

        Returns:
            dets: a float numpy array of shape [n, 5], eg: [[xmin,ymin,xmax,ymax,confidence],[xmin,ymin,xmax,ymax,confidence]]
            lamks: a float numpy array of shape [n, 5, 2]. eg: [[x1,y1],[x2,y2]]
        """
        img = np.float32(img_raw)
        
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        im_height, im_width = img.shape[:2]
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        with torch.no_grad():
            loc, conf, landms = self.model(img)  # forward pass
            # print('det net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.vectorized_forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        landms = landms.reshape((-1, 5, 2))
        # landms = landms.transpose((0, 2, 1))
        # landms = landms.reshape(-1, 10,)
        return dets, landms
