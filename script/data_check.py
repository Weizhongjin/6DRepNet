import sys
import numpy as np
import os
sys.path.append('../')
import src.utils as utils
from PIL import Image
def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print(file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

if __name__ == '__main__':
    filename_path = sys.argv[1]
    data_dir  = sys.argv[2]
    filename_list = get_list_from_filenames(filename_path)
    print(filename_list[0:10])
    while True:
        idx = int(input('index = :'))
        if idx == -1:
            break
        else:
            img = Image.open(os.path.join(data_dir, filename_list[idx] + '.jpg')) 
            img = img.convert('RGB')
            mat_path = os.path.join(data_dir, filename_list[idx] + '.mat')
            pose = utils.get_ypr_from_mat(mat_path)
            pitch = pose[0] * 180 / np.pi
            yaw = pose[1] * 180 / np.pi
            roll = pose[2] * 180 / np.pi
            print('y: {}, p : {} , r : {}'.format(yaw,pitch,roll))
            img.show()