from math import degrees
from matplotlib.pyplot import get
from numpy import angle
from src.utils import get_R
from PIL import Image
from scipy.spatial.transform import Rotation as R
def rotate_img(in_img:Image.Image,in_angles,r_angle):
    '''
    input: 
    1. in_img: origin image
    2. in_angles[p,y,r]: origin pose of face in image
    3. r_angle: rotation angle
    output:
    1. out_img: processed image
    2. out_angle: processed pose of face in image
    other:
    Rotation about the x axis = pitch
    Rotation about the y-axis = yaw
    Rotation about the z-axis = roll
    '''
    out_img = in_img.rotate(-r_angle,expand=True)
    r_ori = R.from_euler('xyz',in_angles,degrees=True)
    r_aug = R.from_euler('xyz',[0,0,r_angle],degrees=True)
    m_ori = r_ori.as_matrix()
    m_aug = r_aug.as_matrix()
    m_new = m_ori.dot(m_aug)
    r_new = R.from_matrix(m_new)
    out_angle = r_new.as_euler('xyz',degrees=True)
    return out_img, out_angle

if __name__ == '__main__':
    img = Image.open('images/y+75;p+5;r-30.jpg')
    in_angle = [75,5,-30]
    print(in_angle)
    out_img, out_angle = rotate_img(img,in_angle,-30)
    print(out_angle)
    out_img.show()

    # angles_1 = [5,75,-30] # [p,y,r]
    # angles_2 = [0,0,30]
    # r = R.from_euler('xyz',[angles_1,angles_2],degrees=True)
    # m_1 = r[0].as_matrix()
    # m_2 = r[1].as_matrix()
    # m = m_1.dot(m_2)
    # r_new = R.from_matrix(m)
    # print(r_new.as_euler('xyz',degrees=True))
    # print(get_R(angles_1[0],angles_1[1],angles_1[2]))
    # print(get_R(angles_2[0],angles_2[1],angles_2[2]))