import cv2
import os
import sys
import pandas as pd
import numpy as np
from face_detection import RetinaFace
from tqdm import tqdm

if __name__ == '__main__':
    model = RetinaFace(gpu_id=-1)
    df = pd.read_csv(sys.argv[1])
    root_path = sys.argv[2]
    output_path = sys.argv[3]
    bboxes = []
    for index,row in tqdm(df.iterrows()):
        img_path = row['path']
        img = cv2.imread(img_path)
        faces = model(img)
        scores = [i[2] for i in faces]
        index = np.argmax(scores)
        bbox = faces[index][0]
        score = scores[index]
        if score < 0.95:
            print(score)
            print(img_path)
        bboxes.append(bbox)
    
    print(len(bboxes))
    bboxes_list = [i.tolist() for i in bboxes]
    df['bboxes'] = bboxes_list
    df.to_csv(output_path,header=True,index=False)