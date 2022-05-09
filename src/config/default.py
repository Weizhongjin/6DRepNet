from yacs.config import CfgNode as CN

_C = CN()
_C.gpu_id = 0
_C.batch_size = 64
_C.num_epochs = 30
_C.lr = 0.00001
_C.snapshot = ''
_C.output_string = '6drepnet'


_C.TRAIN_DATA = CN()
_C.TRAIN_DATA.type = ['Pose_300W_LP']
_C.TRAIN_DATA.data_dir = ['/datasets/300W_LP']
_C.TRAIN_DATA.filename_list = ['datasets/300W_LP/files.txt']
_C.TRAIN_DATA.lp_ratio = 0.1

_C.VAL_DATA = CN()
_C.VAL_DATA.type = ['Pose_300W_LP']
_C.VAL_DATA.data_dir = ['/datasets/300W_LP']
_C.VAL_DATA.filename_list = ['datasets/300W_LP/files.txt']
_C.VAL_DATA.lp_ratio = 0.1
