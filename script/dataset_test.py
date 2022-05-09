import sys
import torch
import torchvision
import yacs
from torchvision import transforms
from torch.utils.data import ConcatDataset,dataloader
sys.path.append('../')
from src.config import _C
from src.datasets import getDataset
if __name__ == '__main__':
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.Resize(240),
                                          transforms.RandomCrop(224),
                                          transforms.ToTensor(),
                                          normalize])
    _C.merge_from_file('../config/data_test.yaml')
    print(_C)
    dataset = getDataset(_C.DATA,transformations)
    print(len(dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=_C.batch_size,
        shuffle=True,
        num_workers=4)
    for data in train_loader:
        print(data[-1])
        break