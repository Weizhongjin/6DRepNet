import sys
import datasets
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import ConcatDataset,dataloader
from config import _C
from datasets import getDataset
if __name__ == '__main__':
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.Resize(240),
                                          transforms.RandomCrop(224),
                                          transforms.ToTensor(),
                                          normalize])
    _C.merge_from_file('config/concat.yaml')
    print(_C)
    dataset = getDataset(_C.DATA,transformations)
    for data in dataset:
        print(data)
        break