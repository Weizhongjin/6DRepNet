from src.sixDRepNet_Estimator import PoseEstimator
import argparse
import cv2

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--image',default='images/temp.jpg',type=str)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    app_name = 'demo'
    pose_estimator = PoseEstimator(model_pth=args.snapshot,app_name=app_name)
    img = cv2.imread(args.image)
    result = pose_estimator.predict(img)
    print(result)
