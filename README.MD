# <div align="center"> **6D Rotation Representation for Unconstrained Head Pose Estimation (Pytorch)** </div>

<p align="center">
  <img src="https://github.com/thohemp/archive/blob/main/6DRepNet2.gif" alt="animated" />
</p>

## <div align="center"> **Paper**</div>
### PDF

### <div align="center"> **Abstract**</div>
In this paper, we present a method for unconstrained end-to-end head pose estimation. We address the problem of ambiguous rotation labels by introducing the rotation matrix formalism for our ground truth data and propose a continuous 6D rotation matrix representation for efficient and robust direct regression. This way, our method can learn the full rotation appearance which is contrary to previous approaches that restrict the pose prediction to a narrow-angle for satisfactory results. In addition, we propose a geodesic distance-based loss to penalize our network with respect to the <img src="https://render.githubusercontent.com/render/math?math=\textit{SO}(3)"> manifold geometry. Experiments on the public AFLW2000 and BIWI datasets demonstrate that our proposed method significantly outperforms other state-of-the-art methods by up to 20\%.
___

|                        |  |<td colspan=4>AFLW2000     |  <td colspan=4>BIWI       |
| --------------------- | -------------- | ------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|                        | Full Range |        Yaw        |  Pitch   |   Roll   |   MAE    |   Yaw    |  Pitch   |   Roll   |   MAE    |
| HopeNet (<img src="https://render.githubusercontent.com/render/math?math=\alpha"> =2) |           N       |       6.47        |   6.56   |   5.44   |   6.16   |   5.17   |   6.98   |   3.39   |   5.18   |
| HopeNet  (<img src="https://render.githubusercontent.com/render/math?math=\alpha"> =1)|           N       |       6.92        |   6.64   |   5.67   |   6.41   |   4.81   |   6.61   |   3.27   |   4.90   |
| FSA-Net                |           N       |       4.50        |   6.08   |   4.64   |   5.07   |   4.27   |   4.96   |   2.76   |   4.00   |
| HPE                    |           N       |       4.80        |   6.18   |   4.87   |   5.28   |   3.12   |   5.18   |   4.57   |   4.29   |
| QuatNet                |          N        |       3.97        |   5.62   |   3.92   |   4.50   | **2.94** |   5.49   |   4.01   |   4.15   |
| WHENet-V               |         N         |       4.44        |   5.75   |   4.31   |   4.83   |   3.60   | **4.10** |   2.73   |   3.48   |
| WHENet                 |         Y/N         |       5.11        |   6.24   |   4.92   |   5.42   |   3.99   |   4.39   |   3.06   |   3.81   |
| TriNet                 |         Y         |       4.04        |   5.77   |   4.20   |   4.67   |   4.11   |   4.76   |   3.05   |   3.97   |
| FDN                    |         N         |       3.78        |   5.61   |   3.88   |   4.42   |   4.52   |   4.70   | **2.56** |   3.93   |
|                        |                  |                   |          |          |          |          |          |          |          |
| **6DRepNet**               |         Y        |     **3.63**      | **4.91** | **3.37** | **3.97** |   3.24   |   4.48   |   2.68   | **3.47** |
|                        |                  |                   |          |          |          |          |          |          |          | |

<div align="center">

## **BIWI 70/30**

|                         |          |          |          |          |
| :---------------------- | :------: | :------: | :------: | :------: |
|                         |   Yaw    |  Pitch   |   Roll   |   MAE    |
| HopeNet (<img src="https://render.githubusercontent.com/render/math?math=\alpha"> =1) |   3.29   |   3.39   |   3.00   |   3.23   |
| FSA-Net                 |   2.89   |   4.29   |   3.60   |   3.60   |
| TriNet                  |   2.93   |   3.04   |   2.44   |   2.80   |
| FDN                     |   3.00   |   3.98   |   2.88   |   3.29   |
|                         |          |          |          |          |
| **6DRepNet**                | **2.69** | **2.92** | **2.36** | **2.66** |
|                         |          |          |          |          |
</div>

---

# <div align="center"> **Quick Start**: </div>

```sh
git clone https://github.com/thohemp/6DRepNet
cd 6DRepNet
```
### Set up a virtual environment:
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  #Install required packages
```
In order to run the demo scripts you need to install the face detector

```sh
pip install git+https://github.com/elliottzheng/face-detection.git@master
```

##  <div align="center"> **Camera Demo**:</div>
Download weights here:

```sh
python demo.py  --snapshot 300W_LP_AFLW2000.tar
                --cam_id 0
                --show_viz False
```


___

# <div align="center"> **Test/Train 3DRepNet** </div>
## <div align="center">  **Preparing datasets** </div>
Download datasets:

* **300W-LP**, **AFLW2000** from [here](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm).

* **BIWI** (Biwi Kinect Head Pose Database) from [here](https://icu.ee.ethz.ch/research/datsets.html) 

Store them in the *datasets* directory.

For 300W-LP and AFLW2000 we need to create a *filenamelist*. 
```
python create_filename_list.py --root_dir datasets/300W_LP
```
The BIWI datasets needs be preprocessed by a face detector to cut our the faces from the images. You can use the script provided [here](https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi.py). For 7:3 splitting of the BIWI dataset you can use the equivilent script [here](https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi_70_30.py). We set the cropped images size to *256*.



## <div align="center"> **Testing**: </div>

### Run test:

```sh
python test.py  --batch_size 64
                --dataset ALFW2000 #BIWI, AFLW2000
                --data_dir datasets/AFLW2000
                --filename_list datasets/AFLW2000/files.txt
                --snapshot output/snapshots/1.tar #Path to weight file 
```

## <div align="center">  **Training** </div>

```sh
python train.py --batch_size 64
                --num_epochs 30
                --lr 0.00001
                --dataset Pose_300W_LP
                --data_dir datasets/300W_LP
                --filename_list datasets/300W_LP/files.txt
```

___
## <div align="center"> **Deploy models** </div>

For reparameterization the trained models into inference-models use the convert script.

```
python convert.py input-model.tar output-model.pth
```

Inference-models are loaded with the flag ```deploy=True```.

```python
model = SixDRepNet(backbone_name='RepVGG-B1g2',
                    backbone_file='',
                    deploy=True,
                    pretrained=False)
```