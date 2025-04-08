#!/usr/bin/python                                                       
# This has been adapted from https://github.com/sairin1202/Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/
# Please refer to the above link for more details or queries.           
                                                                        
import torch
import torchvision
from torchvision.models import vgg16
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor, ToPILImage
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import skimage.io as io

import numpy as np
import glob
import cv2
import itertools
import PIL.Image as Image
import argparse
import os

from image_dataloader_augs import image_loader
#from image_dataloader import image_loader
from utils import *
from network_unet import *

from tqdm import tqdm

print ("Train import done successfully")

# input argmuments
parser = argparse.ArgumentParser(description='SmartKC Mire Segmentation')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.0005,
                    help='weight decay value')
parser.add_argument('--gpu_ids', default=[0], help='a list of gpus')
parser.add_argument('--num_worker', default=4, type=int, help='numbers of worker')
parser.add_argument('--batch_size', default=4, type=int, help='bacth size')
parser.add_argument('--epochs', default=1, type=int, help='epochs')
parser.add_argument('--start_epochs', default=0, type=int, help='start epochs')


parser.add_argument('--image_size', default = 512, type=int, help='size of input image frame')
parser.add_argument('--train_data', help='training data directory')
parser.add_argument('--val_data', help='validation data directory')
parser.add_argument('--train_txt', help='training image pair names txt')
parser.add_argument('--val_txt', help='validation image pair names txt')
parser.add_argument('--train_label', help='training label directory')
parser.add_argument('--val_label', help='validation label directory')
parser.add_argument('--model_path',type=str, help='model saving directory')
parser.add_argument('--output_dir', default = 'output', type=str, help='model saving directory')

args = parser.parse_args()


# let the label pixels =1 if it >0
class Relabel:
    def __call__(self, tensor):
        assert isinstance(
            tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor > 0] = 1
        return tensor

# numpy -> tensor
class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()



class Trainer:
    def __init__(self):
        self.args = args
        self.input_transform = Compose([Resize((args.image_size, args.image_size)), ToTensor(
        ), Normalize([.485, .456, .406], [.229, .224, .225])])

        self.label_transform = None

        os.makedirs(args.model_path, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.output_dir+'/train', exist_ok=True)
        os.makedirs("stats", exist_ok=True)

        # loading checkpoint
        #checkpoint = torch.load('epoch_37_iter_200.pkl')
        self.net = model().cuda()
        self.net = nn.DataParallel(self.net, device_ids=self.args.gpu_ids)
        #self.net.load_state_dict(checkpoint)

        self.train_data_loader = DataLoader(image_loader(self.args.train_data, self.args.train_label, self.args.train_txt, args.image_size, self.input_transform, self.label_transform),
                                            num_workers=self.args.num_worker, batch_size=self.args.batch_size, shuffle=True)
        self.val_data_loader = DataLoader(image_loader(self.args.val_data, self.args.val_label, self.args.val_txt, args.image_size, self.input_transform, self.label_transform, "test"),
                                          num_workers=self.args.num_worker, batch_size=self.args.batch_size, shuffle=False)
        #self.optimizer = optim.Adam(
        #    self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # if using pre-trained VGG make trainable first few layers as false
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=0.9)
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.loss_func = nn.CrossEntropyLoss()

    def evaluate(self, net, epoch, iteration):

        self.net.eval()
        correct = 0
        wrong = 0
        intersection = 0
        union = 0
        true_positive = 0
        positive = 1
        
        if iteration%200 == 0:
            os.system('mkdir '+args.output_dir+'/'+str(epoch)+'_'+str(iteration))
        
        test_losses = []
        for i, (image1, label1) in tqdm(enumerate(self.val_data_loader)):
            
            image1, label1 = image1.cuda(), label1.cuda()
            output = self.net(image1)

            # calculate loss only from output2
            loss = self.loss_func(output, label1)
            test_losses.append(loss.data.cpu().numpy())
            
            output = torch.argmax(output, dim=1)

            if iteration%200 == 0:
                save_inter(image1, label1, output, str(epoch)+'_'+str(iteration), i, 2, args.output_dir)

            output = output.cuda()
            # for ROI = class label 3
            #output[output == 1] = 0
            #output[output == 3] = 0
            #output[output == 2] = 1
            #label1[label1 == 1] = 0
            #label1[label1 == 3] = 0
            #label1[label1 == 2] = 1

            # eval output2
            c, w = pixel_accuracy(output, label1)
            correct += c
            wrong += w

            i, u = jaccard(output, label1)
            intersection += i
            union += u

            tp, p = precision(output, label1)
            true_positive += tp
            positive += p


        print("pixel accuracy: {} correct: {}  wrong: {}".format(
            correct / (correct + wrong), correct, wrong))
        print("precision: {} true_positive: {} positive: {}".format(
            true_positive / positive, true_positive, positive))
        print("jaccard score: {} intersection: {} union: {}".format(
            intersection / union, intersection, union))
        print("epoch {} iter {}/{} Validation Output BCE loss: {}".format(epoch,
            iteration, len(self.val_data_loader), np.mean(test_losses)))
        self.net.train()

        return correct / (correct + wrong), intersection / union, true_positive / positive

    def train(self):
        best_acc = -1
        for _, epoch in tqdm(enumerate(range(self.args.start_epochs, self.args.epochs))):
            losses = []
            for i, (image1, label1) in tqdm(enumerate(self.train_data_loader)):
                
                image1, label1 = image1.cuda(), label1.cuda()
                output = self.net(image1)

                # calculate loss from output2
                loss = self.loss_func(output, label1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.data.cpu().numpy())

                if i == len(self.train_data_loader)-1:
                    print("---------------------------------------------")
                    print("epoch {} iter {}/{} Train Total loss: {} Best ACC Yet: {}".format(epoch,
                        i, len(self.train_data_loader), np.mean(losses), best_acc))
                    acc, jac, pre = self.evaluate(self.net, epoch, i)
                    save_stats(epoch, i, [np.mean(losses)], "train", "stats")
                    if best_acc < acc:
                        best_acc = acc
                        torch.save(self.net.state_dict(),
                                args.model_path+'/best_epoch_'+str(epoch)+'_iter_'+str(i)+'.pkl')
                        print("************************BEST ACC ACHIEVED**********************")
                        print("\n\n")
                if i == len(self.train_data_loader)-1 and i != 0:
                    torch.save(self.net.state_dict(),
                               args.model_path+'/epoch_'+str(epoch)+'_iter_'+str(i)+'.pkl')
            if epoch > 500:
                self.exp_lr_scheduler.step()
                print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
