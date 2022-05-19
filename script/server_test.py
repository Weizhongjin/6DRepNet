import sys
from turtle import left
sys.path.append('../')
from src.server.client_com import ImageServerClient
from src.utils import draw_axis
import cv2
import numpy as np
def draw_from_res(res,img):
    npimage = np.fromstring(img, np.uint8)
    img_ori = cv2.imdecode(npimage, cv2.IMREAD_COLOR)
    img_plot = img_ori
    for temp_res in res:
        angles = temp_res['pose']
        bboxs = temp_res['location']
        # translation = temp_res['translation']
        translation = [int((bboxs['left']*2+bboxs['width'])/2),int((bboxs['top']*2+bboxs['height'])/2)]
        axis_size = int((bboxs['width']+bboxs['height'])/2)
        img_plot = draw_axis(img_plot, angles['yaw'], angles['pitch'],
            angles['roll'], translation[0], translation[1], size = axis_size)
        # img_plot = cv2.rectangle(img_plot,(int(bboxs[0]),int(bboxs[1])),(int(bboxs[2]),int(bboxs[3]),(0,255,0),2)
    cv2.imwrite('client_output.jpg',img_plot)
if __name__ == '__main__':
    img_path = sys.argv[1] 
    d = ImageServerClient('http://127.0.0.1:8080/faceAngle')
    # d = ImageServerClient('http://aiipgateway.jiutian.hq.cmcc/facerec/dev/pose/inference')
    image = open(img_path, 'rb').read()
    idict = d.encode_warpimg(image)
    idict['image_name'] = img_path
    req = d.send(idict)
    res = req['result_data']['face_info']
    print(res)
    draw_from_res(res,image)
