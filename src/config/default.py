from yacs.config import CfgNode as CN

_C = CN()
_C.DATA = CN()
_C.gpu_id = 0
_C.batch_size = 64
_C.num_epochs = 30
_C.lr = 0.00001
_C.snapshot = ''
_C.output_string = '6drepnet'
_C.DATA.type = ['Pose_300W_LP']
_C.DATA.data_dir = ['/datasets/300W_LP']
_C.DATA.filename_list = ['datasets/300W_LP/files.txt']