import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
print('# GPUs = %d' % (torch.cuda.device_count()))

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/public/st/ParC-Net")
sys.path.append("/home/public/st/")
import json

import numpy as np
import torchvision.datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
#from art.utils import load_cifar10
from load_data import load_cifar10
#from networks import *
import resnet
#import wide_resnet
#import wideresnet
#import vgg
#import inceptionv3
#import densenet
#import shufflenetv2
import generate_and_test
import generate_and_test_new

from arf.Arf_MobileNetV1 import arf_mv1

model = arf_mv1.cuda()
batch_size = 64
use_cuda = True
device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
# 设置模型路径
#pretrained_target = "/home/public/st/adversarial_robustness_toolbox_main/examples/CIFAR10_256.pth"
pretrained_target = "/home/public/st/adversarial_robustness_toolbox_main/models_test/arf_mv1_ori.pth"
# pretrained_target = './model/cifar10_wide_resnet.pth'
# pretrained_target = './model/model_cifar_wrn.pt'

#attack_type = 'pgd'
#attack_type = 'cw'
#attack_type = 'MIFGSM'
#attack_type = 'SparseFool'
#attack_type = 'BIM'
#attack_type = 'RFGSM'
#attack_type = 'DeepFool'
#attack_type = 'APGD'
#attack_type = 'OnePixel'
#attack_type = 'Jitter'
attack_type = 'Auto'
#attack_type = 'PGD'
#attack_type = 'DIFGSM'
#attack_type = 'FAB'
#attack_type = 'pgd'
#attack_type = 'PGDL2'
#attack_type = 'CW'
#attack_type = 'deepfool'
#attack_type = 'fgsm'
#attack_type = 'FGSM'
#attack_type = 'AA'
#attack_type = 'Wasserstein'
#attack_type = 'GeoDA'
#attack_type = 'Square'
#attack_type = 'SquareAttack'
#attack_type = 'TIFGSM'

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

def train_target_model(target_model, epochs, train_dataloader, test_dataloader, dataset_size, model_name):
    train_num = 50000
    print('train_num is ',train_num)
    
    target_model.train()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)

    #generate_and_test.generate(target_model, attack_type)
    generate_and_test_new.generate(target_model, attack_type)
    #torch.save(target_model.state_dict(), "../models/{}_{}_0.pth".format(attack_type, model_name))

    for epoch in range(0,epochs):
        loss_epoch = 0

        for i, data in enumerate(train_dataloader, 0):

            train_imgs, train_labels = data
            #print("train_imgs:", train_imgs)
            train_imgs = generate_and_test_new.generate_and_return(train_imgs, train_labels, target_model, attack_type)
            #train_imgs = generate_and_test.generate_and_return(train_imgs, train_labels, target_model, attack_type)
            # train_imgs = torch.from_numpy(train_imgs)
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            #print(train_imgs.shape)
            
            logits_model = target_model(train_imgs)
            criterion = F.cross_entropy(logits_model, train_labels)
            loss_epoch += criterion
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()

        print('Loss in epoch {}: {}'.format(epoch, loss_epoch.item()))

        # 每隔5个epoch，使用测试集检测成功率
        if epoch % 5 == 0 or epoch < 5:
            #generate_and_test.generate(target_model, attack_type)
            #generate_and_test_new.generate(target_model, attack_type)
            torch.save(target_model.state_dict(), "../models/{}_{}_{}_{}_origin_001.pth".format(train_num, attack_type, model_name, epoch))
            #torch.save(target_model.state_dict(), "../models/{}_{}_{}_origin_001.pth".format(train_num, model_name, epoch))
            
    # save model
    targeted_model_file_name = '../models/{}_{}_{}_origin_lr001.pth'.format(train_num, attack_type, model_name)
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()
     
    #target_model.load_state_dict(torch.load("/data/st/adversarial-robustness-toolbox-main/models/10000_pgd_resnet_95.pth", map_location={'cuda:1':'cuda:2'}))
    # 测试集准确率
    n_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        #print("shape:",test_img.shape)
        #print(test_img)
        test_img, test_label = test_img.to(device), test_label.to(device)
        
        pred_lab = torch.argmax(target_model(test_img), 1)
        #print(pred_lab)
        #print(test_label)
        n_correct += torch.sum(pred_lab == test_label,0)
    
    print(dataset_size)
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy in test set: {}%\n'.format(100 * n_correct.item()/dataset_size))

    n_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img = generate_and_test.generate_and_return(test_img, test_label, target_model, attack_type)
        #print("shape:",test_img.shape)
        #print(test_img)
        test_img, test_label = test_img.to(device), test_label.to(device)
        
        pred_lab = torch.argmax(target_model(test_img), 1)
        print(pred_lab)
        print(test_label)
        n_correct += torch.sum(pred_lab == test_label,0)
    
    print(dataset_size)
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy in test set: {}%\n'.format(100 * n_correct.item()/dataset_size))

#print("Load resnet model")
model.load_state_dict(torch.load(pretrained_target, map_location={'cuda:1':'cuda:2'}))
#model.load_state_dict(torch.load("/home/public/st/adversarial_robustness_toolbox_main/models/50000_parc_convnext_origin_001_8159.pth", map_location={'cuda:1':'cuda:2'}))

print("load train dataset")

x_train = []
y_train = []
name_to_label = {"airplane":0 , "automobile":1 ,"bird":2, "cat":3 ,"deer":4 , "dog":5, "frog":6 ,"horse":7, "ship":8 ,"truck":9}
label_to_name = {0:"airplane" , 1:"automobile" , 2:"bird", 3:"cat" , 4:"deer" , 5:"dog", 6:"frog" , 7:"horse", 8:"ship", 9:"truck"}

#(x_train_origin, y_train_origin), (x_test, y_test), (y_train_normal, y_test_normal), min_pixel_value, max_pixel_value = load_cifar10()

#x_train_origin = np.transpose(x_train_origin, (0, 3, 1, 2)).astype(np.float32)


trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
#train_dataset = MyDataset(data=x_train, labels=y_train, transform=trans)
#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
train_dataset = torchvision.datasets.CIFAR10('./datasets', train=True, transform=trans, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

print("load test dataset")
test_dataset = torchvision.datasets.CIFAR10('./datasets', train=False, transform=trans, download=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
print("start to train attack model")
train_target_model(model, 200, train_dataloader, test_dataloader, len(test_dataset), model_name)
