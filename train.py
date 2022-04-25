import sys
import os
import argparse
import time

import numpy as np

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

from src.model import SixDRepNet, SixDRepNet2
import src.datasets as datasets
from src.loss import GeodesicLoss

import torch.utils.model_zoo as model_zoo
import torchvision
from src.config import _C

def get_ignored_params(model):
    b = [model.layer0]
    #b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model):
    b = [model.linear_reg]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':

    cfg_file = sys.argv[1]
    _C.merge_from_file(cfg_file)
    cudnn.enabled = True
    num_epochs = _C.num_epochs
    batch_size = _C.batch_size
    gpu = _C.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    summary_name = '{}_{}_bs{}'.format(
        'SixDRepNet', int(time.time()), _C.batch_size)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))

    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='model/RepVGG-B1g2-train.pth',
                        deploy=False,
                        pretrained=True)
    if not _C.snapshot == '':
        saved_state_dict = torch.load(_C.snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.Resize(240),
                                          transforms.RandomCrop(224),
                                          transforms.ToTensor(),
                                          normalize])

    pose_dataset = datasets.getDataset(
        _C.DATA, transformations)

    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    model.cuda(gpu)
    crit =  GeodesicLoss().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)

    optimizer = torch.optim.Adam([
        {'params': get_ignored_params(model), 'lr': 0},
        {'params': get_non_ignored_params(model), 'lr': _C.lr},
        {'params': get_fc_params(model), 'lr': _C.lr * 10}
    ], lr=_C.lr)

    if not _C.snapshot == '':
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])

    #milestones = np.arange(num_epochs)
    milestones = [10, 20]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    print('Starting training.')
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0
        for i, (images, gt_mat, _, _) in enumerate(train_loader):
            iter += 1
            images = torch.Tensor(images).cuda(gpu)

            # Forward pass
            pred_mat = model(images)

            # Calc loss
            loss = crit(gt_mat.cuda(gpu), pred_mat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if (i+1) % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: '
                      '%.6f' % (
                          epoch+1,
                          num_epochs,
                          i+1,
                          len(pose_dataset)//batch_size,
                          loss.item(),
                      )
                      )

        scheduler.step()

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...',
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                  }, 'output/snapshots/' + summary_name + '/' + _C.output_string +
                      '_epoch_' + str(epoch+1) + '.tar')
                  )
