# -*- coding: utf-8 -*-
"""
@Time ： 2021/9/14 11:08
@Auth ： shangjunfeng
@mail ： shangjunfeng@chinamobile.com
@File ：face_dection.py
"""
# here put the import lib

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from .face_detect.detector import face_det_model

import cv2






class FaceDetection():
    def __init__(self,face_model=''):

        self.face_detector = face_det_model(face_model)
        print('Load Face Dection Finish:', face_model)
    def face_dec(self, img):

        bboxes,bounding_boxes,points = self.face_detector.detect_multi_face(img)

        if len(bboxes) == 0:
            print('no det face...')
            return None, []

        batch_imgs = []
        batch_boxs = []
        batch_poiont = []
        # get all boxes
        for crop_face, bbox,point in bboxes:
            batch_imgs.append(crop_face)
            batch_boxs.append(bbox[:4])
            batch_poiont.append(point)
        return bounding_boxes,points

    def dect_face(self,img):
        faces,landmarks = self.face_dec(img)

        DetResult_list = []
        if faces is not None:
            for i in range(faces.shape[0]):

                requests = {
                    "XMin": None,
                    "YMax": None,
                    "XMax": None,
                    "YMin": None,
                }

                requests["XMin"] = float(faces[i][0])
                requests["YMin"] = float(faces[i][1])
                requests["XMax"] = float(faces[i][2])
                requests["YMax"] = float(faces[i][3])
                requests["score"] = float(faces[i][4])
                # requests["landMarks"] = landmarks[i].tolist()

                DetResult_list.append(requests)

        return DetResult_list
