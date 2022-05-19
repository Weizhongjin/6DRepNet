import os
import sys
import pandas as pd
import cv2
from tqdm import tqdm
sys.path.append('../')
from src.sixDRepNet_Estimator import PoseEstimator

if __name__ == '__main__':
    csv_path = sys.argv[1]
    root_path = sys.argv[2]
    pose_esm = PoseEstimator(app_name='pose eva')
    # d = ImageServerClient('http://aiipgateway.jiutian.hq.cmcc/facerec/dev/pose/inference')
    df = pd.read_csv(csv_path)
    y_err = 0
    p_err = 0
    r_err = 0
    for index, row in df.iterrows():
        print('==========process {}==========='.format(index))
        img_path = row['path']
        img_path = os.path.join(root_path,img_path)
        image = cv2.imread(img_path)
        res = pose_esm.predict(image)
        angles = res[0]['pose']
        y = row['y']
        p = row['p']
        r = row['r']
        y_err += abs(y-angles['yaw'])
        p_err += abs(p-angles['pitch'])
        r_err += abs(r-angles['roll'])
        print('target:{}'.format([y,p,r]))
        print('predict:{}'.format(angles))
    mae = (y_err+p_err+r_err)/3
    print('yaw err: {}; pitch err: {}; row err: {}; mae : {}'.format(y_err/180,p_err/180,r_err/180,mae/180))    
