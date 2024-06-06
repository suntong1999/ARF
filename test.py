from utils.data_loader import get_data_loader
from models import wgan_gradient_penalty
from lib.transforms import Pad, Crop
import os
import resnet
import numpy as np
import time
import sys

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
print('# GPUs = %d' % (torch.cuda.device_count()))
import torchvision

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

import numpy as np

import time

import torchvision.datasets
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from load_data import load_cifar10

import generate_and_test_new
import resnet_cifar100
import resnetcifar

import resnet
from cifar100_model import ResNet34
from arf.Arf_MobileNetV1 import arf_mv1
import generate_and_test
import generate_and_test_new
from trans import Pad, Crop
model_name = 'resnet34'

from common import load_state_dict,Opencv2PIL,TorchMeanStdNormalize

use_cuda = True
device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')

model = arf_mv1.cuda()
batch_size = 64
pretrained_target = "/data/ltx/adversarial_detector/pre_trained/arf_mv1.pth"

attack_type = 'pgd'

#img_dir = "/data/st/pytorch-wgan-master/np_data_cifar10_gan_origin_No_t/50_60_1000"

# filenames是训练数据文件名称列表，labels是标签列表
class MyDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

 
    def __len__(self):
        return self.data.shape[0]
 
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        # return image, self.labels[idx]
        return x, y

if __name__ == '__main__':
    print('\nCHECKING FOR CUDA...')
    use_cuda = True
    print('CUDA Available: ',torch.cuda.is_available())
    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
#    model = resnet_cifar100.ResNet34(num_c=100).to(device)
    #model = resnetcifar.resnet34cifar(num_classes=100)
    nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
            11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,
            21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,
            31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,
            41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,
            51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,
            61 ,62 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70 ,
            71 ,72 ,73 ,74 ,75 ,76 ,77 ,78 ,79 ,80 ,
            81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,
            91 ,92 ,93 ,94 ,95 ,96 ,97 ,98 ,99]
    labels = ["0","1","2","3" ,"4", "5", "6" , "7", "8", "9","10",
              "11","12","13" ,"14", "15", "16" , "17", "18", "19","20",
              "21","22","23" ,"24", "25", "26" , "27", "28", "29","30",
              "31","32","33" ,"34", "35", "36" , "37", "38", "39","40",
              "41","42","43" ,"44", "45", "46" , "47", "48", "49","50",
              "51","52","53" ,"54", "55", "56" , "57", "58", "59","60",
              "61","62","63" ,"64", "65", "66" , "67", "68", "69","70",
              "71","72","73" ,"74", "75", "76" , "77", "78", "79","80",
              "81","82","83" ,"84", "85", "86" , "87", "88", "89","90",
              "91","92","93" ,"94", "95", "96" , "97", "98", "99"
              ]
              
    load_state_dict(pretrained_target, model)
#    model.load_state_dict(torch.load("/data/ltx/adversarial_detector/pre_trained/resnet_cifar100.pth", map_location="cpu"))
#    model.load_state_dict(torch.load("/data/st/gan-cifar100/resnet34cifar-acc77.840.pth", map_location="cpu"))
    
    print("load generate test dataset")
    
    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        
    ])
    
    
    
    img_dir = "/data/st/diffusers-main/merge_npy/"
    x_train = []
    y_train = []
    
    files = os.listdir(img_dir)
    for file in files:
        name = file.split('.')[0]
        data = np.load(img_dir + '/' + file).astype(np.float32)
        label = [int(name)] * len(data)
        print(len(data))
        x_train.extend(data)
        y_train.extend(label)
    
    
    x_train = np.array(x_train)
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    y_train = np.array(y_train)
    
    
    
    #trans = transforms.Compose([
    #    transforms.Resize(32),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #])
    in_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), Pad(),
                                            Crop(crop_type='random', crop_frac=0.8), ])
    #train_dataset = torchvision.datasets.CIFAR100('./datasets', train=True, transform=in_transform, download=True)
    train_dataset_AUG = MyDataset(data=x_train, labels=y_train, transform=in_transform)
    train_dataloader_AUG = torch.utils.data.DataLoader(train_dataset_AUG, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
#    in_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), Pad(),
#                                        Crop(crop_type='random', crop_frac=0.8), ])
#    in_transform = transform=transforms.Compose([
##                      Opencv2PIL(),
#                        transforms.ToTensor(),
#                      transforms.Normalize(mean=np.array([125.3, 123.0, 113.9]) / 255.0,
#                                            std=np.array([63.0, 62.1, 66.7]) / 255.0),
#            ])
    train_dataset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=in_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
#    in_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), Pad(),
    testset = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=in_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)

    
#    generate_and_test_new.generate(model, attack_type)

    model.eval()
    
    
    print('original model direct test:')
    # 测试集准确率
    n_correct = 0
    for i, data in enumerate(train_dataloader_AUG, 0):
        test_img, test_label = data
        print(test_img.shape)
        test_img, test_label = test_img.to(device), test_label.to(device)
        model = model.to(device)
        pred_lab = torch.argmax(model(test_img), 1)
        n_correct += torch.sum(pred_lab == test_label,0)
    
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy in train aug set: {}%\n'.format(100 * n_correct.item()/100000))
    
    n_correct = 0
    for i, data in enumerate(train_dataloader, 0):
        test_img, test_label = data
        print(test_img.shape)
        test_img, test_label = test_img.to(device), test_label.to(device)
        model = model.to(device)
        pred_lab = torch.argmax(model(test_img), 1)
        n_correct += torch.sum(pred_lab == test_label,0)
    
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy in train ori set: {}%\n'.format(100 * n_correct.item()/50000))

    # 测试集对抗准确率
#    n_correct = 0
#    for i, data in enumerate(testloader, 0):
#        test_img, test_label = data
#        adv_img = generate_and_test_new.generate_and_return(test_img, test_label, model, attack_type)
#        adv_img, test_label = adv_img.to(device), test_label.to(device)
#        
#        pred_lab = torch.argmax(model(adv_img), 1)
#        n_correct += torch.sum(pred_lab == test_label,0)
#    
#    print('Correctly Classified: ', n_correct.item())
#    print('Accuracy in adv test set: {}%\n'.format(100 * n_correct.item()/10000))

    
  
        
    
        
